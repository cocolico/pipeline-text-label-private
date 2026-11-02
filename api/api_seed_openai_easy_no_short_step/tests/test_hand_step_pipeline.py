import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hand_step_pipeline as hand_steps
import stage1_generate_draft as stage1


class HandStepPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = {
            "total_frames": 100,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "duration_sec": 100 / 30.0,
        }

    def test_vlm_events_require_inclusive_zero_based_ordered_intervals(self) -> None:
        payload, errors = hand_steps._normalize_vlm_events(
            {
                "events": [
                    {
                        "event_id": 1,
                        "start_frame_id": 10,
                        "end_frame_id": 20,
                        "active_hands": ["right_hand"],
                        "action_description": "The right hand lifts the cup.",
                        "confidence": 0.9,
                    }
                ]
            },
            total_frames=100,
        )
        self.assertEqual(errors, [])
        self.assertEqual(payload["events"][0]["start_frame_id"], 10)
        self.assertEqual(payload["events"][0]["end_frame_id"], 20)

    def test_stage1_semantics_reject_non_english_descriptions(self) -> None:
        normalized, errors = hand_steps._normalize_step_semantics(
            {
                "step_id": 1,
                "action_semantics": {
                    "left_hand": "左手拿起抹布",
                    "right_hand": "not active",
                    "overall": "The left hand picks up the cloth.",
                },
            },
            step_id=1,
        )
        self.assertIsNone(normalized)
        self.assertTrue(any("left_hand must be pure English" in error for error in errors))

    def test_stage1_semantics_accept_pure_english_descriptions(self) -> None:
        normalized, errors = hand_steps._normalize_step_semantics(
            {
                "step_id": 1,
                "action_semantics": {
                    "left_hand": "The left hand picks up the cloth.",
                    "right_hand": "not active",
                    "overall": "The left hand picks up the green cloth.",
                },
            },
            step_id=1,
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized["right_hand"], "not active")

    def test_object_reid_preserves_entity_ref_after_temporary_loss(self) -> None:
        step = {"hand_step_id": 2, "start_frame_id": 100, "end_frame_id": 160}
        inventory = [{"entity_ref": "step_002_obj_001", "canonical_name": "green cloth"}]
        normalized, errors = hand_steps._normalize_object_reidentification(
            {
                "step_id": 2,
                "objects": [{
                    "entity_ref": "step_002_obj_001",
                    "canonical_name": "green cloth",
                    "continuity_status": "reacquired",
                    "disappearance_intervals": [{
                        "last_visible_frame_id": 120,
                        "first_reappeared_frame_id": 132,
                    }],
                    "reappearance_detections": [{
                        "frame_id": 132,
                        "bbox_xyxy": [20, 30, 80, 100],
                        "match_confidence": 0.91,
                        "identity_evidence": "The color, folded edge, and hand interaction match the earlier cloth.",
                    }],
                }],
            },
            step=step,
            object_inventory=inventory,
            width=1920,
            height=1080,
        )
        self.assertEqual(errors, [])
        self.assertTrue(normalized[0]["identity_preserved_within_step"])
        self.assertEqual(normalized[0]["reappearance_detections"][0]["bbox_source"], "vlm_reid_redetection")

    def test_object_reid_rejects_reacquisition_without_redetection_bbox(self) -> None:
        normalized, errors = hand_steps._normalize_object_reidentification(
            {
                "step_id": 1,
                "objects": [{
                    "entity_ref": "step_001_obj_001",
                    "canonical_name": "cup",
                    "continuity_status": "reacquired",
                    "disappearance_intervals": [{
                        "last_visible_frame_id": 10,
                        "first_reappeared_frame_id": 15,
                    }],
                    "reappearance_detections": [],
                }],
            },
            step={"hand_step_id": 1, "start_frame_id": 0, "end_frame_id": 30},
            object_inventory=[{"entity_ref": "step_001_obj_001", "canonical_name": "cup"}],
            width=640,
            height=480,
        )
        self.assertIsNone(normalized)
        self.assertTrue(any("requires a bbox" in error for error in errors))

    def test_bbox_arm_visibility_accepts_complete_inclusive_intervals(self) -> None:
        step = {
            "hand_step_id": 1,
            "start_frame_id": 10,
            "end_frame_id": 15,
            "bbox_tracks": {
                "left_hand": [
                    {"frame_id": frame_id, "bbox_xyxy": [1, 2, 30, 40], "bbox_source": "upstream_sam3"}
                    for frame_id in range(10, 16)
                ],
                "right_hand": [],
            },
        }
        normalized, errors = hand_steps._normalize_bbox_arm_intervals(
            [{
                "hand": "left_hand",
                "bbox_source": "upstream_sam3",
                "arm_visible_intervals": [
                    {"start_frame_id": 11, "end_frame_id": 12},
                    {"start_frame_id": 14, "end_frame_id": 15},
                ],
            }],
            step=step,
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized[0]["arm_visible_intervals"], [
            {"start_frame_id": 11, "end_frame_id": 12},
            {"start_frame_id": 14, "end_frame_id": 15},
        ])

    def test_bbox_arm_visibility_clips_and_splits_around_missing_bbox_frames(self) -> None:
        normalized, errors = hand_steps._normalize_bbox_arm_intervals(
            [{
                "hand": "left_hand", "bbox_source": "upstream_sam3",
                "arm_visible_intervals": [{"start_frame_id": 10, "end_frame_id": 12}],
            }],
            step={
                "hand_step_id": 1, "start_frame_id": 10, "end_frame_id": 12,
                "bbox_tracks": {
                    "left_hand": [
                        {"frame_id": 10, "bbox_source": "upstream_sam3"},
                        {"frame_id": 12, "bbox_source": "upstream_sam3"},
                    ],
                    "right_hand": [],
                },
            },
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized[0]["arm_visible_intervals"], [
            {"start_frame_id": 10, "end_frame_id": 10},
            {"start_frame_id": 12, "end_frame_id": 12},
        ])

    def test_long_step_video_fps_is_capped_by_sample_budget(self) -> None:
        request_fps = hand_steps._step_video_request_fps(
            {"start_frame_id": 0, "end_frame_id": 3371},
            source_fps=30.0,
            configured_fps=5.0,
        )
        self.assertAlmostEqual(request_fps, 120.0 / (3372.0 / 30.0))
        self.assertLess(request_fps, 1.1)

    def test_compact_stage3_bbox_prompt_keeps_ranges_and_samples_only(self) -> None:
        step = {
            "stage3_bbox_tracks": {
                "left_hand": [
                    {
                        "frame_id": frame_id,
                        "bbox_xyxy": [frame_id, 2, frame_id + 10, 30],
                        "bbox_source": "upstream_sam3",
                    }
                    for frame_id in list(range(0, 50)) + list(range(60, 100))
                ],
                "right_hand": [],
            },
        }
        compact = hand_steps._compact_stage3_bbox_tracks_for_prompt(step)
        left = compact["left_hand"][0]
        self.assertEqual(left["available_frame_intervals"], [
            {"start_frame_id": 0, "end_frame_id": 49},
            {"start_frame_id": 60, "end_frame_id": 99},
        ])
        self.assertLessEqual(len(left["representative_bboxes"]), 12)

    def test_stage1_single_call_writes_all_three_stage_step_outputs(self) -> None:
        api_cfg = type("ApiCfg", (), {"video_fps": 10.0, "max_tokens": 12000})()
        response = (
            '{"step_id":1,"action_semantics":{"left_hand":"lifting",'
            '"right_hand":"not active","overall":"The left hand lifts a cup."},'
            '"multiple_action_subjects":false,"hand_object_relations":[],"object_inventory":[],"bbox_arm_visibility":[]}'
        )
        one_frame_meta = {**self.meta, "total_frames": 30, "duration_sec": 1.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "stage1"), exist_ok=True)
            with patch.object(hand_steps, "get_video_metadata", return_value=one_frame_meta), patch.object(
                hand_steps,
                "load_upstream_hand_data",
                return_value={
                    "available": True,
                    "path": "/tmp/upstream.json",
                    "schema": "x",
                    "frames": {
                        frame_id: {"keep": True, "hands": {}}
                        for frame_id in range(30)
                    },
                },
            ), patch.object(
                hand_steps,
                "call_chat_completion",
                return_value=(response, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
            ) as model_call, patch.object(
                hand_steps,
                "_ensure_step_clip",
                return_value=os.path.join(tmpdir, "stage1", "step_clips", "step_001.mp4"),
            ), patch.object(
                hand_steps,
                "_ensure_step_visualization",
                return_value=os.path.join(tmpdir, "stage1", "step_clips", "step_001_visualization.mp4"),
            ), patch.object(
                hand_steps,
                "_finalize_step_clip_name",
                return_value=os.path.join(
                    tmpdir, "stage1", "step_clips",
                    "step_001_source_frames_0_29_The_left_hand_lifts_a_cup.mp4",
                ),
            ):
                payload, _usage = hand_steps.generate_stage1_vlm_hand_events(
                    video_path="/tmp/rgb.mp4",
                    video_out=tmpdir,
                    client=object(),
                    api_cfg=api_cfg,
                    frames=[{"source_frame_0based": 0}],
                    overwrite=True,
                    max_retries=1,
                )
                stage2_result = hand_steps.read_json(
                    os.path.join(tmpdir, "stage2", "step_001", "object_inventory.json")
                )
                stage3_result = hand_steps.read_json(
                    os.path.join(tmpdir, "stage3", "step_001", "bbox_arm_visibility.json")
                )
                self.assertEqual(model_call.call_count, 1)
                self.assertEqual(model_call.call_args.kwargs["video_fps"], 10.0)
        self.assertEqual(payload["schema"], "upstream_keep_step_semantics.v1")
        self.assertEqual(payload["step_semantics"][0]["start_frame_id"], 0)
        self.assertEqual(payload["step_semantics"][0]["end_frame_id"], 29)
        self.assertEqual(payload["step_semantics"][0]["action_semantics"]["left_hand"], "lifting")
        self.assertEqual(payload["step_semantics"][0]["source_frame_id_offset"], 0)
        self.assertIn("The_left_hand_lifts_a_cup", payload["step_semantics"][0]["clip_path"])
        self.assertEqual(stage2_result["generated_by"], "single_pass_step_video_analysis")
        self.assertEqual(stage3_result["generated_by"], "single_pass_step_video_analysis")
        self.assertEqual(stage3_result["bbox_arm_visibility"], [])
        self.assertEqual(stage3_result["evaluation_scope"], "valid_keep_true_actions_only")
        self.assertEqual(stage3_result["valid_action_intervals"], [
            {"action_run_id": 1, "start_frame_id": 0, "end_frame_id": 29},
        ])

    def test_identity_inventory_reacquires_changed_object_and_derives_missing_gap(self) -> None:
        raw = [{
            "entity_ref": "step_001_obj_001",
            "canonical_name": "plastic bag",
            "description": "A transparent plastic bag with a red label.",
            "selection_reason": "The right hand carries it out and returns with it opened.",
            "priority": "direct_hand_interaction",
            "aliases": ["bag", "plastic bag", "bag"],
            "identity_profile": {
                "stable_features": {
                    "category": "bag", "material": "plastic", "base_color": "transparent",
                    "distinctive_features": ["red label"],
                },
                "mutable_features": ["position", "orientation", "shape", "open_closed_state"],
            },
            "visible_intervals": [
                {"start_frame_id": 100, "end_frame_id": 180},
                {"start_frame_id": 231, "end_frame_id": 340},
            ],
            "representative_frame_ids": [120, 250],
            "hand_interactions": [{
                "hand": "right_hand", "action": "carries out and returns",
                "start_frame_id": 170, "end_frame_id": 250,
            }],
            "state_transitions": [{
                "frame_id": 231, "from": "closed", "to": "opened",
                "cause": "the right hand opens it while partly out of view",
            }],
            "identity_evidence": "The red label, material and continuous right-hand action match.",
            "identity_status": "resolved",
            "confidence": 0.94,
        }]
        inventory, errors = hand_steps._normalize_step_identity_inventory(
            raw, step_id=1, start_frame_id=100, end_frame_id=380
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]["entity_ref"], "step_001_obj_001")
        self.assertEqual(inventory[0]["temporarily_missing_intervals"], [
            {"start_frame_id": 181, "end_frame_id": 230},
        ])
        self.assertEqual(inventory[0]["aliases"], ["bag", "plastic bag"])

    def test_identity_inventory_rejects_unobserved_representative_frame(self) -> None:
        raw = [{
            "entity_ref": "step_001_obj_001", "canonical_name": "cup",
            "description": "A cup.", "selection_reason": "Held by the left hand.",
            "priority": "direct_hand_interaction", "aliases": [],
            "identity_profile": {"stable_features": {"category": "cup"}, "mutable_features": ["position"]},
            "visible_intervals": [{"start_frame_id": 10, "end_frame_id": 20}],
            "representative_frame_ids": [25], "hand_interactions": [], "state_transitions": [],
            "identity_evidence": "", "identity_status": "resolved", "confidence": 0.8,
        }]
        _inventory, errors = hand_steps._normalize_step_identity_inventory(
            raw, step_id=1, start_frame_id=0, end_frame_id=29
        )
        self.assertTrue(any("not inside a visible interval" in error for error in errors))

    def test_inventory_validation_keeps_same_name_instances_separate(self) -> None:
        inventory = [
            {"entity_ref": "step_001_obj_001", "canonical_name": "white bowl", "identity_status": "resolved"},
            {"entity_ref": "step_001_obj_002", "canonical_name": "white bowl", "identity_status": "resolved"},
        ]
        validation = hand_steps._build_step_inventory_validation(inventory, [])
        self.assertEqual(len(validation["same_name_instance_risks"]), 1)
        self.assertEqual(validation["status"], "passed")

    def test_inventory_validation_reports_unknown_relation_reference(self) -> None:
        validation = hand_steps._build_step_inventory_validation(
            [{"entity_ref": "step_001_obj_001", "canonical_name": "cup", "identity_status": "resolved"}],
            [{"entity_ref": "step_001_obj_999"}],
        )
        self.assertEqual(validation["status"], "needs_review")
        self.assertEqual(validation["missing_relation_entity_refs"], ["step_001_obj_999"])

    def test_no_effective_action_writes_empty_stage1_manifests_without_model_call(self) -> None:
        meta = {**self.meta, "total_frames": 29, "duration_sec": 29 / 30.0}
        upstream = {
            "available": True,
            "path": "/tmp/upstream.json",
            "schema": "x",
            "frames": {
                frame_id: {"keep": True, "hands": {}}
                for frame_id in range(29)
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            hand_steps, "get_video_metadata", return_value=meta
        ), patch.object(
            hand_steps, "load_upstream_hand_data", return_value=upstream
        ), patch.object(hand_steps, "call_chat_completion") as model_call:
            clips = hand_steps.materialize_stage1_keep_clips(
                video_path="/tmp/raw_data/rgb.mp4", video_out=tmpdir
            )
            payload, usage = hand_steps.generate_stage1_vlm_hand_events(
                video_path="/tmp/raw_data/rgb.mp4",
                video_out=tmpdir,
                client=None,
                api_cfg=type("ApiCfg", (), {"video_fps": 5.0, "max_tokens": 10000})(),
                frames=[],
                overwrite=True,
                max_retries=1,
            )
        self.assertEqual(clips["status"], "no_effective_action")
        self.assertEqual(clips["steps"], [])
        self.assertEqual(payload["status"], "no_effective_action")
        self.assertEqual(payload["step_semantics"], [])
        self.assertEqual(usage["api_calls"], 0)
        empty_draft = stage1._draft_from_step_semantics(payload)
        self.assertEqual(empty_draft["steps"], [])
        self.assertEqual(stage1._stage1_raw_schema_errors(empty_draft), [])
        self.assertEqual(stage1._draft_hard_errors(empty_draft), [])
        model_call.assert_not_called()

    def test_step_clip_uses_inclusive_source_interval_and_reuses_output(self) -> None:
        step = {"hand_step_id": 4, "start_frame_id": 30, "end_frame_id": 59}
        with tempfile.TemporaryDirectory() as tmpdir:
            def fake_cut(**kwargs):
                self.assertEqual(kwargs["start_frame_id"], 30)
                self.assertEqual(kwargs["end_frame_id"], 59)
                self.assertAlmostEqual(kwargs["fps"], 30.0)
                os.makedirs(os.path.dirname(kwargs["output_path"]), exist_ok=True)
                with open(kwargs["output_path"], "wb") as handle:
                    handle.write(b"clip")

            with patch.object(hand_steps, "_cut_exact_source_frames", side_effect=fake_cut) as cut:
                first = hand_steps._ensure_step_clip(
                    video_path="/tmp/rgb.mp4", video_out=tmpdir, step=step, fps=30.0
                )
                second = hand_steps._ensure_step_clip(
                    video_path="/tmp/rgb.mp4", video_out=tmpdir, step=step, fps=30.0
                )
        self.assertEqual(first, second)
        self.assertEqual(cut.call_count, 1)
        self.assertIn(os.path.join("stage1", "step_clips"), first)

    def test_stage1_filters_short_runs_and_merges_across_short_total_gap(self) -> None:
        upstream = {
            "available": True,
            "path": "/tmp/hand_screening.json",
            "schema": "x",
            "frames": {
                **{frame_id: {"keep": True, "hands": {}} for frame_id in range(0, 40)},
                **{frame_id: {"keep": False, "hands": {}} for frame_id in range(40, 50)},
                **{frame_id: {"keep": True, "hands": {}} for frame_id in range(50, 55)},
                **{frame_id: {"keep": False, "hands": {}} for frame_id in range(55, 65)},
                **{frame_id: {"keep": True, "hands": {}} for frame_id in range(65, 105)},
            },
        }
        clip_meta = {**self.meta, "total_frames": 105, "duration_sec": 3.5}
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            hand_steps, "get_video_metadata", return_value=clip_meta
        ), patch.object(
            hand_steps, "load_upstream_hand_data", return_value=upstream
        ), patch.object(
            hand_steps, "_ensure_step_clip",
            side_effect=lambda **kw: os.path.join(
                tmpdir, "stage1", "step_clips",
                f"step_{kw['step']['hand_step_id']:03d}.mp4",
            ),
        ) as cut:
            payload = hand_steps.materialize_stage1_keep_clips(
                video_path="/tmp/raw_data/rgb.mp4", video_out=tmpdir
            )
            saved = hand_steps.read_json(os.path.join(tmpdir, "stage1", "step_clips.json"))
        self.assertEqual(
            [(x["start_frame_id"], x["end_frame_id"]) for x in payload["steps"]],
            [(0, 104)],
        )
        self.assertEqual([x["frame_count"] for x in payload["steps"]], [105])
        self.assertEqual(cut.call_count, 1)
        self.assertEqual(saved["minimum_valid_keep_frames"], 30)
        self.assertEqual(saved["merge_when_intervening_frames_less_than"], 90)
        self.assertEqual(saved["candidate_frame_ranges"]["valid_keep_ranges"], [
            {"start_frame_id": 0, "end_frame_id": 39},
            {"start_frame_id": 65, "end_frame_id": 104},
        ])
        self.assertEqual(saved["candidate_frame_ranges"]["merged_step_ranges"], [
            {"start_frame_id": 0, "end_frame_id": 104},
        ])
        self.assertEqual(saved["steps"][0]["valid_action_intervals"], [
            {"action_run_id": 1, "start_frame_id": 0, "end_frame_id": 39},
            {"action_run_id": 2, "start_frame_id": 65, "end_frame_id": 104},
        ])

    def test_step_clip_prompt_preserves_original_frame_mapping(self) -> None:
        text = hand_steps._step_clip_mapping_text(
            {"start_frame_id": 120, "end_frame_id": 179}
        )
        self.assertIn("original source frames 120 through 179", text)
        self.assertIn("original frame_id=120+n", text)
        self.assertIn("never clip-local IDs", text)

    def test_stage2_and_stage3_resolve_clip_from_stage1_manifest(self) -> None:
        step = {"hand_step_id": 2, "start_frame_id": 120, "end_frame_id": 179}
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_rel = os.path.join(
                "stage1", "step_clips",
                "step_002_source_frames_120_179_The_right_hand_lifts_the_cup.mp4",
            )
            clip_abs = os.path.join(tmpdir, clip_rel)
            os.makedirs(os.path.dirname(clip_abs), exist_ok=True)
            with open(clip_abs, "wb") as handle:
                handle.write(b"video")
            hand_steps.write_json(os.path.join(tmpdir, "stage1", "step_clips.json"), {
                "steps": [{
                    "step_id": 2, "start_frame_id": 120, "end_frame_id": 179,
                    "clip_path": clip_rel,
                }]
            })
            resolved = hand_steps._require_stage1_step_clip(video_out=tmpdir, step=step)
        self.assertEqual(resolved, clip_abs)

    def test_stage2_loads_single_pass_result_without_model_client(self) -> None:
        boundaries = {"hand_steps": [{"hand_step_id": 1}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            step_dir = os.path.join(tmpdir, "step_001")
            os.makedirs(step_dir)
            hand_steps.write_json(os.path.join(step_dir, "object_inventory.json"), {
                "step_id": 1,
                "hand_object_relations": [],
                "object_inventory": [{"entity_ref": "step_001_obj_001"}],
                "generated_by": "single_pass_step_video_analysis",
            })
            with patch.object(
                hand_steps, "initialize_api_client",
                side_effect=AssertionError("Stage 2 must not initialize a model client"),
            ):
                result = hand_steps._generate_stage2_step_objects(
                    video_path="/tmp/rgb.mp4",
                    boundaries=boundaries,
                    api_cfg=object(),
                    max_retries=1,
                    stage2_dir=tmpdir,
                )
        self.assertEqual(result[1]["object_inventory"][0]["entity_ref"], "step_001_obj_001")

    def test_step_clip_name_appends_sanitized_overall_after_end_frame(self) -> None:
        path = hand_steps._step_clip_path(
            "/tmp/output", 2, 120, 179, "The left hand picks up / the green cloth."
        )
        self.assertTrue(
            path.endswith("step_002_source_frames_120_179_The_left_hand_picks_up_the_green_cloth.mp4")
        )

    def test_stage1_draft_contains_hand_descriptions_frames_and_renamed_clip(self) -> None:
        payload = {"step_semantics": [{
            "step_id": 1,
            "start_frame_id": 120,
            "end_frame_id": 179,
            "clip_path": "stage1/step_clips/step_001_source_frames_120_179_The_right_hand_lifts_the_cup.mp4",
            "action_semantics": {
                "left_hand": "The left hand is not active.",
                "right_hand": "The right hand lifts the cup.",
                "overall": "The right hand lifts the cup.",
            },
        }]}
        draft = stage1._draft_from_step_semantics(payload)
        item = draft["steps"][0]
        self.assertEqual(item["step_goal"], item["overall"])
        self.assertEqual(item["start_frame_id"], 120)
        self.assertEqual(item["end_frame_id"], 179)
        self.assertEqual(item["left_hand"], "The left hand is not active.")
        self.assertIn("179_The_right_hand_lifts_the_cup.mp4", item["clip_path"])
        self.assertEqual(stage1._stage1_raw_schema_errors(draft), [])
        self.assertEqual(stage1._draft_hard_errors(draft), [])

    def test_upstream_frame_ids_expand_by_half_second_and_clamp(self) -> None:
        selected = hand_steps._expanded_upstream_frame_ids(
            {"frames": {
                2: {"keep": True, "hands": {}},
                25: {"keep": False, "hands": {}},
                50: {"keep": False, "hands": {"left_hand": {"detected": True, "bbox_xyxy": [1, 2, 3, 4]}}},
                98: {"keep": True, "hands": {}},
            }},
            total_frames=100,
            fps=30.0,
            padding_sec=0.5,
        )
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 99)
        self.assertTrue(set(range(35, 66)).issubset(selected))
        self.assertNotIn(25, selected)

    def test_short_keep_run_is_not_a_step(self) -> None:
        upstream = {
            "available": True,
            "schema": "hand_frame_screening.v2",
            "path": "/tmp/hand_screening.json",
            "frames": {
                2: {
                    "keep": False,
                    "hands": {
                        "left_hand": {"detected": True, "bbox_xyxy": [10, 10, 30, 30]},
                        "right_hand": {"detected": False, "bbox_xyxy": None},
                    },
                },
                3: {
                    "keep": True,
                    "hands": {
                        "left_hand": {"detected": True, "bbox_xyxy": [30, 10, 50, 30]},
                        "right_hand": {"detected": True, "bbox_xyxy": [40, 10, 60, 30]},
                    },
                },
                22: {
                    "keep": False,
                    "hands": {
                        "left_hand": {"detected": False, "bbox_xyxy": None},
                        "right_hand": {"detected": True, "bbox_xyxy": [80, 80, 120, 120]},
                    },
                },
                50: {
                    "keep": False,
                    "hands": {
                        "left_hand": {"detected": False, "bbox_xyxy": None},
                        "right_hand": {"detected": True, "bbox_xyxy": [100, 100, 140, 140]},
                    },
                },
            },
        }
        vlm = {
            "events": [
                {
                    "event_id": 1,
                    "start_frame_id": 20,
                    "end_frame_id": 25,
                    "active_hands": ["right_hand"],
                    "action_description": "The right hand presses a control.",
                    "confidence": 0.8,
                }
            ]
        }
        result = hand_steps.build_hand_step_boundaries(
            video_path="/tmp/raw_data/rgb.mp4",
            vlm_events=vlm,
            upstream=upstream,
            meta=self.meta,
            max_gap_frames=0,
        )
        covered = set()
        for step in result["hand_steps"]:
            covered.update(step["frame_ids"])
        self.assertEqual(result["hand_steps"], [])
        self.assertEqual(covered, set())
        self.assertEqual(result["upstream"]["left_detected_frame_count"], 2)
        self.assertEqual(result["upstream"]["right_detected_frame_count"], 3)
        upstream_bboxes = [
            bbox
            for step in result["hand_steps"]
            for track in step["bbox_tracks"].values()
            for bbox in track
        ]
        self.assertEqual(upstream_bboxes, [])

    def test_exactly_30_keep_frames_is_valid(self) -> None:
        upstream = {
            "available": True,
            "frames": {
                frame_id: {"keep": True, "hands": {}}
                for frame_id in range(10, 40)
            },
        }
        result = hand_steps.build_hand_step_boundaries(
            video_path="/tmp/raw_data/rgb.mp4", vlm_events={}, upstream=upstream,
            meta=self.meta,
        )
        self.assertEqual(
            [(x["start_frame_id"], x["end_frame_id"]) for x in result["hand_steps"]],
            [(10, 39)],
        )

    def test_gap_of_exactly_90_frames_does_not_merge(self) -> None:
        upstream = {
            "available": True,
            "frames": {
                **{frame_id: {"keep": True, "hands": {}} for frame_id in range(0, 40)},
                **{frame_id: {"keep": False, "hands": {}} for frame_id in range(40, 130)},
                **{frame_id: {"keep": True, "hands": {}} for frame_id in range(130, 160)},
            },
        }
        meta = {**self.meta, "total_frames": 160, "duration_sec": 160 / 30.0}
        result = hand_steps.build_hand_step_boundaries(
            video_path="/tmp/raw_data/rgb.mp4", vlm_events={}, upstream=upstream,
            meta=meta,
        )
        self.assertEqual(
            [(x["start_frame_id"], x["end_frame_id"]) for x in result["hand_steps"]],
            [(0, 39), (130, 159)],
        )

    def test_gap_of_89_frames_merges(self) -> None:
        upstream = {
            "available": True,
            "frames": {
                **{frame_id: {"keep": True, "hands": {}} for frame_id in range(0, 40)},
                **{frame_id: {"keep": False, "hands": {}} for frame_id in range(40, 129)},
                **{frame_id: {"keep": True, "hands": {}} for frame_id in range(129, 159)},
            },
        }
        meta = {**self.meta, "total_frames": 159, "duration_sec": 159 / 30.0}
        result = hand_steps.build_hand_step_boundaries(
            video_path="/tmp/raw_data/rgb.mp4", vlm_events={}, upstream=upstream,
            meta=meta,
        )
        self.assertEqual(
            [(x["start_frame_id"], x["end_frame_id"]) for x in result["hand_steps"]],
            [(0, 158)],
        )

    def test_stage3_bbox_tracks_exclude_merged_bridge_and_short_action_frames(self) -> None:
        frames = {
            **{frame_id: {"keep": True, "hands": {}} for frame_id in range(0, 40)},
            **{frame_id: {"keep": False, "hands": {}} for frame_id in range(40, 50)},
            **{frame_id: {"keep": True, "hands": {}} for frame_id in range(50, 55)},
            **{frame_id: {"keep": False, "hands": {}} for frame_id in range(55, 65)},
            **{frame_id: {"keep": True, "hands": {}} for frame_id in range(65, 105)},
        }
        for frame_id in (5, 45, 52, 70):
            frames[frame_id]["hands"] = {
                "left_hand": {"detected": True, "bbox_xyxy": [1, 2, 20, 30]},
            }
        upstream = {"available": True, "frames": frames}
        meta = {**self.meta, "total_frames": 105, "duration_sec": 3.5}
        result = hand_steps.build_hand_step_boundaries(
            video_path="/tmp/raw_data/rgb.mp4", vlm_events={}, upstream=upstream,
            meta=meta,
        )
        step = result["hand_steps"][0]
        self.assertEqual(step["valid_action_intervals"], [
            {"action_run_id": 1, "start_frame_id": 0, "end_frame_id": 39},
            {"action_run_id": 2, "start_frame_id": 65, "end_frame_id": 104},
        ])
        self.assertEqual(
            [item["frame_id"] for item in step["bbox_tracks"]["left_hand"]],
            [5, 45, 52, 70],
        )
        self.assertEqual(
            [item["frame_id"] for item in step["stage3_bbox_tracks"]["left_hand"]],
            [5, 70],
        )
        normalized, errors = hand_steps._normalize_bbox_arm_intervals(
            [{
                "hand": "left_hand",
                "bbox_source": "upstream_sam3",
                "arm_visible_intervals": [
                    {"start_frame_id": 5, "end_frame_id": 5},
                    {"start_frame_id": 70, "end_frame_id": 70},
                ],
            }],
            step=step,
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized[0]["arm_visible_intervals"], [
            {"start_frame_id": 5, "end_frame_id": 5},
            {"start_frame_id": 70, "end_frame_id": 70},
        ])

    def test_review_score_uses_minimum_participating_hand(self) -> None:
        step = {
            "hand_step_id": 7,
            "start_frame_id": 10,
            "end_frame_id": 20,
            "active_hands": ["left_hand", "right_hand"],
        }
        obj = {
            "step_id": 7,
            "action_semantics": {
                "left_hand": "The left hand holds the bowl.",
                "right_hand": "The right hand stirs the food.",
                "overall": "The left hand holds the bowl while the right hand stirs.",
            },
            "hand_object_relations": [
                {"object": "bowl", "hand": "left hand", "relation": "holding"}
            ],
            "visibility": {
                "left_hand": {"arm_visible": 0, "hand_completeness_score": 0.61},
                "right_hand": {"arm_visible": 1, "hand_completeness_score": 0.88},
            },
            "frame_visibility": [
                {
                    "frame_id": frame_id,
                    "left_hand": {"arm_visible": 0, "hand_completeness_score": 0.61},
                    "right_hand": {"arm_visible": 1, "hand_completeness_score": 0.88},
                }
                for frame_id in (10, 20)
            ],
            "fallback_bboxes": [],
        }
        normalized, errors = hand_steps._normalize_review(
            obj,
            step=step,
            width=1920,
            height=1080,
            representative_frame_ids=[10, 20],
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized["visibility"]["overall_hand_completeness_score"], 0.61)
        self.assertEqual(normalized["visibility"]["arm_visible_any"], 1)

    def test_stage3_rejects_new_fallback_bbox_because_stage1_is_authoritative(self) -> None:
        step = {
            "hand_step_id": 1,
            "start_frame_id": 0,
            "end_frame_id": 10,
            "active_hands": ["left_hand"],
        }
        obj = {
            "step_id": 1,
            "action_semantics": {
                "left_hand": "The left hand taps the button.",
                "right_hand": "not active",
                "overall": "The left hand taps the button.",
            },
            "hand_object_relations": [],
            "visibility": {
                "left_hand": {"arm_visible": 0, "hand_completeness_score": 0.5},
                "right_hand": {"arm_visible": 0, "hand_completeness_score": 0.0},
            },
            "frame_visibility": [
                {
                    "frame_id": 5,
                    "left_hand": {"arm_visible": 0, "hand_completeness_score": 0.5},
                    "right_hand": {"arm_visible": 0, "hand_completeness_score": 0.0},
                }
            ],
            "fallback_bboxes": [
                {
                    "frame_id": 5,
                    "hand": "left_hand",
                    "bbox_xyxy": [1, 2, 30, 40],
                    "bbox_source": "vlm_grounding",
                }
            ],
        }
        normalized, errors = hand_steps._normalize_review(
            obj,
            step=step,
            width=1920,
            height=1080,
            representative_frame_ids=[5],
        )
        self.assertIsNone(normalized)
        self.assertTrue(any("fallback_bboxes must be empty" in error for error in errors))

    def test_stage3_requests_vlm_bbox_even_when_upstream_bbox_exists(self) -> None:
        step = {
            "hand_step_id": 1,
            "start_frame_id": 3,
            "end_frame_id": 3,
            "active_hands": ["left_hand"],
            "draft_action_semantics": "The left hand lifts a cup.",
            "bbox_tracks": {
                "left_hand": [
                    {"frame_id": 3, "bbox_xyxy": [10, 10, 30, 30], "bbox_source": "upstream"}
                ],
                "right_hand": [],
            },
        }

        class FakeCapture:
            def isOpened(self):
                return True

            def set(self, *_args):
                return True

            def read(self):
                return True, object()

            def release(self):
                return None

        with patch.object(hand_steps.cv2, "VideoCapture", return_value=FakeCapture()), patch.object(
            hand_steps, "_representative_frame_ids", return_value=[3]
        ), patch.object(hand_steps, "_encode_crop", return_value="encoded"):
            content, manifest = hand_steps.build_hand_step_roi_content(
                video_path="/tmp/rgb.mp4",
                step=step,
                meta={"width": 100, "height": 80},
            )

        self.assertNotIn("request_vlm_generated_bboxes", manifest[0])
        image_items = [item for item in content if item.get("type") == "image_url"]
        # Complete frame for VLM grounding + expanded semantic ROI + exact SAM3 bbox.
        self.assertEqual(len(image_items), 3)
        prompt = hand_steps.build_hand_step_review_prompt(step, manifest, 100, 80)
        self.assertIn("Do not generate bboxes here", prompt)

    def test_bbox_evaluations_keep_vlm_and_sam3_scores_separate(self) -> None:
        expected = [
            {"frame_id": 3, "hand": "left_hand", "bbox_source": "upstream_sam3", "bbox_xyxy": [1, 2, 20, 30]},
            {"frame_id": 3, "hand": "left_hand", "bbox_source": "vlm_generated", "bbox_xyxy": [2, 3, 22, 32]},
        ]
        normalized, errors = hand_steps._normalize_bbox_evaluations(
            {"evaluations": [
                {"frame_id": 3, "hand": "left_hand", "bbox_source": "upstream_sam3", "arm_visible": 1, "hand_completeness_score": 0.75},
                {"frame_id": 3, "hand": "left_hand", "bbox_source": "vlm_generated", "arm_visible": 0, "hand_completeness_score": 0.45},
            ]},
            expected=expected,
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized[0]["hand_completeness_score"], 0.75)
        self.assertEqual(normalized[1]["hand_completeness_score"], 0.45)

    def test_stage2_references_stage1_clips_without_creating_new_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            unit_dir = os.path.join(tmpdir, "camera_1_rgb_depth_001")
            raw_dir = os.path.join(unit_dir, "raw_data")
            upstream_dir = os.path.join(unit_dir, "processed_hand_screening")
            output_root = os.path.join(tmpdir, "processed_text_label")
            video_path = os.path.join(raw_dir, "rgb.mp4")
            os.makedirs(raw_dir)
            os.makedirs(upstream_dir)
            with open(video_path, "wb") as fh:
                fh.write(b"placeholder")
            hand_steps.write_json(
                os.path.join(upstream_dir, "hand_screening.json"),
                {
                    "schema": "hand_frame_screening.v2",
                    "frames": [
                        *[
                            {"frame_index": frame_id, "keep": True, "left_hand": {"detected": True, "bbox_xyxy": [1, 2, 20, 30]}, "right_hand": {"detected": False, "bbox_xyxy": None}}
                            for frame_id in range(3, 33)
                        ],
                        *[
                            {"frame_index": frame_id, "keep": True, "left_hand": {"detected": False, "bbox_xyxy": None}, "right_hand": {"detected": True, "bbox_xyxy": [40, 5, 60, 35]}}
                            for frame_id in range(183, 213)
                        ],
                    ],
                },
            )
            stage1_dir = os.path.join(output_root, "rgb", "stage1")
            hand_steps.write_json(
                os.path.join(stage1_dir, "vlm_hand_events.json"),
                {"events": []},
            )
            expected_clips = []
            for step_id, start, end in ((1, 3, 32), (2, 183, 212)):
                clip_path = hand_steps._step_clip_path(
                    os.path.join(output_root, "rgb"), step_id, start, end
                )
                os.makedirs(os.path.dirname(clip_path), exist_ok=True)
                with open(clip_path, "wb") as handle:
                    handle.write(b"stage1 clip")
                expected_clips.append(clip_path)
            def fake_object_maps(**kwargs):
                steps = kwargs["boundaries"]["hand_steps"]
                self.assertEqual(
                    [item["frame_ids"] for item in steps],
                    [list(range(3, 33)), list(range(183, 213))],
                )
                self.assertEqual(steps[0]["bbox_tracks"]["left_hand"][0]["bbox_xyxy"], [1, 2, 20, 30])
                return {
                    step["hand_step_id"]: {
                        "hand_object_relations": [],
                        "object_inventory": [{"entity_ref": f"step_{step['hand_step_id']:03d}_obj_001"}],
                    }
                    for step in steps
                }

            stage2_meta = {**self.meta, "total_frames": 213, "duration_sec": 213 / 30.0}
            with patch.object(hand_steps, "get_video_metadata", return_value=stage2_meta):
                with patch.object(hand_steps, "_generate_stage2_step_objects", side_effect=fake_object_maps) as generate:
                    hand_steps.run_stage2_text_only(
                        video_path=video_path,
                        output_root=output_root,
                        overwrite=True,
                        api_cfg=object(),
                    )
            self.assertEqual(generate.call_count, 1)
            stage2_dir = os.path.join(output_root, "rgb", "stage2")
            self.assertTrue(os.path.exists(os.path.join(stage2_dir, "hand_step_boundaries.json")))
            self.assertTrue(os.path.exists(os.path.join(stage2_dir, "step_segments.json")))
            boundaries = hand_steps.read_json(os.path.join(stage2_dir, "hand_step_boundaries.json"))
            self.assertEqual(len(boundaries["hand_steps"]), 2)
            self.assertEqual(boundaries["hand_steps"][1]["object_inventory"][0]["entity_ref"], "step_002_obj_001")
            segments = hand_steps.read_json(os.path.join(stage2_dir, "step_segments.json"))
            self.assertTrue(all(item["clip_materialized"] for item in segments["segments"]))
            self.assertTrue(all(item["clip_relpath"].startswith("stage1") for item in segments["segments"]))
            media = []
            for root, _dirs, names in os.walk(output_root):
                media.extend(os.path.join(root, name) for name in names if name.lower().endswith((".mp4", ".jpg", ".png")))
            self.assertEqual(sorted(media), sorted(expected_clips))


    def test_temporarily_shrunken_bbox_is_repaired_without_mutating_source(self) -> None:
        tracks = {"left_hand": [], "right_hand": []}
        for frame_id in range(61):
            bbox = [100, 100, 220, 260]
            if frame_id in (29, 30, 31):
                bbox = [145, 170, 175, 210]
            tracks["left_hand"].append({
                "frame_id": frame_id, "bbox_xyxy": list(bbox), "bbox_source": "upstream_sam3"
            })
        original = [list(item["bbox_xyxy"]) for item in tracks["left_hand"]]
        analysis, annotations = hand_steps._repair_temporarily_shrunken_bboxes(
            tracks, width=640, height=480
        )
        self.assertEqual([item["bbox_xyxy"] for item in tracks["left_hand"]], original)
        repaired = {item["frame_id"]: item for item in analysis["left_hand"]}
        self.assertEqual(repaired[30]["pre_expansion_analysis_bbox_xyxy"], [100, 100, 220, 260])
        self.assertEqual(repaired[30]["analysis_bbox_xyxy"], [88, 84, 232, 276])
        self.assertTrue(repaired[30]["bbox_repair_applied"])
        self.assertTrue(any(item["frame_id"] == 30 and item["possible_wrist_only"] for item in annotations))

    def test_every_analysis_bbox_gets_deterministic_glove_margin_without_changing_source(self) -> None:
        tracks = {
            "left_hand": [{"frame_id": 5, "bbox_xyxy": [100, 80, 200, 180], "bbox_source": "upstream_sam3"}],
            "right_hand": [],
        }
        analysis, annotations = hand_steps._repair_temporarily_shrunken_bboxes(
            tracks, width=640, height=480
        )
        self.assertEqual(annotations, [])
        self.assertEqual(tracks["left_hand"][0]["bbox_xyxy"], [100, 80, 200, 180])
        self.assertEqual(analysis["left_hand"][0]["analysis_bbox_xyxy"], [90, 70, 210, 190])
        self.assertTrue(analysis["left_hand"][0]["bbox_expansion_applied"])

    def test_sustained_natural_scale_change_is_not_repaired_without_larger_box_afterward(self) -> None:
        tracks = {"left_hand": [], "right_hand": []}
        for frame_id in range(61):
            scale = 1.0 - 0.5 * frame_id / 60.0
            bbox = [100, 100, 100 + int(round(120 * scale)), 100 + int(round(160 * scale))]
            tracks["left_hand"].append({
                "frame_id": frame_id, "bbox_xyxy": bbox, "bbox_source": "upstream_sam3"
            })
        analysis, annotations = hand_steps._repair_temporarily_shrunken_bboxes(
            tracks, width=640, height=480
        )
        self.assertEqual(annotations, [])
        self.assertTrue(all(not item["bbox_repair_applied"] for item in analysis["left_hand"]))

    def test_arm_visibility_marks_temporarily_small_source_bbox_as_unreliable(self) -> None:
        step = {
            "hand_step_id": 1,
            "start_frame_id": 10,
            "end_frame_id": 12,
            "stage3_bbox_tracks": {
                "left_hand": [
                    {"frame_id": frame_id, "bbox_xyxy": [1, 2, 20, 30], "bbox_source": "upstream_sam3"}
                    for frame_id in (10, 11, 12)
                ],
                "right_hand": [],
            },
            "bbox_quality_annotations": [{
                "frame_id": 11,
                "hand": "left_hand",
                "bbox_source": "upstream_sam3",
                "source_bbox_may_not_contain_complete_hand": True,
            }],
        }
        normalized, errors = hand_steps._normalize_bbox_arm_intervals(
            [{
                "hand": "left_hand",
                "bbox_source": "upstream_sam3",
                "arm_visible_intervals": [{"start_frame_id": 10, "end_frame_id": 12}],
            }],
            step=step,
        )
        self.assertEqual(errors, [])
        self.assertTrue(normalized[0]["bbox_may_not_contain_complete_hand"])
        self.assertTrue(normalized[0]["result_may_be_unreliable"])
        self.assertEqual(normalized[0]["unreliable_frame_intervals"], [{"start_frame_id": 11, "end_frame_id": 11}])


if __name__ == "__main__":
    unittest.main()
