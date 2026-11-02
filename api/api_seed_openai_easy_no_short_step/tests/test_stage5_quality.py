from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_stage5():
    common = types.ModuleType("common")
    for name in (
        "add_api_cli_args",
        "api_config_from_args",
        "build_source_video_meta",
        "ensure_video_out_dir_safe",
        "format_duration",
        "get_video_metadata",
        "now_utc_iso",
        "read_json",
        "update_run_summary",
        "video_id_from_path",
        "write_json",
    ):
        setattr(common, name, lambda *args, **kwargs: None)
    common.logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    common.DEFAULT_ARK_REASONING_EFFORT = "medium"
    spec = importlib.util.spec_from_file_location(
        "stage5_quality_test_target", ROOT / "stage5_bbox_tracking.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"common": common, spec.name: module}):
        spec.loader.exec_module(module)
    return module


M = _load_stage5()


def detection(box, confidence=0.8, source="test"):
    return {
        "bbox": list(box),
        "confidence": float(confidence),
        "source": source,
        "bbox_extent": "observed_full",
    }


class PrimaryOperatorSelectionTests(unittest.TestCase):
    def test_center_pair_beats_higher_confidence_edge_pair(self):
        frames = []
        for frame_idx in range(12):
            candidates = {
                "left_hand": [detection([15, 230, 95, 350], 0.99, "edge_left")],
                "right_hand": [detection([110, 230, 190, 350], 0.98, "edge_right")],
            }
            if frame_idx not in {0, 1, 6}:
                candidates["left_hand"].append(
                    detection([390 + frame_idx, 250, 480 + frame_idx, 380], 0.58, "primary_left")
                )
                candidates["right_hand"].append(
                    detection([520 + frame_idx, 245, 615 + frame_idx, 380], 0.57, "primary_right")
                )
            frames.append(candidates)

        records, qa = M._select_primary_operator_sequence(frames, 1000, 600)
        self.assertFalse(records[0])
        self.assertFalse(records[1])
        self.assertFalse(records[6])
        for frame_idx in set(range(12)) - {0, 1, 6}:
            self.assertEqual(records[frame_idx]["left_hand"]["source"], "primary_left")
            self.assertEqual(records[frame_idx]["right_hand"]["source"], "primary_right")
        self.assertEqual(qa["discarded_secondary_candidate_boxes"], 24)

    def test_temporal_identity_lock_rejects_transient_center_impostor(self):
        frames = []
        for frame_idx in range(8):
            primary_left = detection([360 + frame_idx, 250, 440 + frame_idx, 370], 0.70, "primary_left")
            primary_right = detection([480 + frame_idx, 250, 560 + frame_idx, 370], 0.70, "primary_right")
            candidates = {
                "left_hand": [primary_left],
                "right_hand": [primary_right],
            }
            if frame_idx == 4:
                candidates["left_hand"].append(
                    detection([440, 245, 500, 365], 0.99, "impostor_left")
                )
                candidates["right_hand"].append(
                    detection([505, 245, 565, 365], 0.99, "impostor_right")
                )
            frames.append(candidates)

        records, qa = M._select_primary_operator_sequence(frames, 1000, 600)
        self.assertEqual(records[4]["left_hand"]["source"], "primary_left")
        self.assertEqual(records[4]["right_hand"]["source"], "primary_right")
        self.assertGreaterEqual(qa["pair_identity_lock_frames"], 1)

    def test_single_hand_global_reference_rejects_edge_substitution(self):
        frames = []
        for frame_idx in range(8):
            candidates = {
                "left_hand": [detection([10, 240, 90, 360], 0.99, "edge_left")]
            }
            if frame_idx >= 2:
                candidates["left_hand"].append(
                    detection([450, 240, 530, 360], 0.60, "primary_left")
                )
            frames.append(candidates)
        records, qa = M._select_primary_operator_sequence(frames, 1000, 600)
        self.assertFalse(records[0])
        self.assertFalse(records[1])
        self.assertEqual(qa["reference_mode"], "single_hand_fallback")
        for frame_idx in range(2, 8):
            self.assertEqual(records[frame_idx]["left_hand"]["source"], "primary_left")

    def test_solo_abrupt_secondary_excursion_is_left_missing_not_switched(self):
        frames = []
        for frame_idx in range(9):
            if frame_idx in {3, 4, 5}:
                box = [760, 0, 900, 150]
                source = "secondary_edge_right"
            else:
                box = [430 + frame_idx * 4, 250, 530 + frame_idx * 4, 390]
                source = "primary_right"
            frames.append({"right_hand": [detection(box, 0.8, source)]})

        records, qa = M._select_primary_operator_sequence(frames, 1000, 600)
        for frame_idx in {3, 4, 5}:
            self.assertNotIn("right_hand", records[frame_idx])
        self.assertGreaterEqual(qa["rejected_temporal_identity_switches"], 3)


class HandRecoveryTests(unittest.TestCase):
    def test_border_reentry_allows_large_area_change_without_relaxing_center_motion(self):
        plausible, metrics = M._hand_transition_is_plausible(
            [990, 0, 999, 20], [850, 0, 980, 180], 2, 1000, 600
        )
        self.assertTrue(plausible)
        self.assertGreater(metrics["area_ratio"], 6.0)

        far_plausible, _ = M._hand_transition_is_plausible(
            [990, 0, 999, 20], [100, 0, 260, 180], 2, 1000, 600
        )
        self.assertFalse(far_plausible)

    def test_same_locked_sam3_instance_allows_coherent_fast_motion(self):
        previous = [400, 250, 500, 390]
        current = [560, 180, 660, 320]
        strict, _ = M._hand_transition_is_plausible(
            previous, current, 2, 1000, 600
        )
        locked, _ = M._locked_sam3_instance_transition_is_plausible(
            previous, current, 2, 1000, 600
        )
        self.assertFalse(strict)
        self.assertTrue(locked)

    def test_short_gap_fill_does_not_bridge_far_identity(self):
        records = [
            {"left_hand": detection([10, 200, 90, 320])},
            {},
            {"left_hand": detection([850, 200, 930, 320])},
        ]
        M._fill_short_hand_gaps(records, 3)
        self.assertNotIn("left_hand", records[1])

    def test_short_gap_fill_interpolates_plausible_motion(self):
        records = [
            {"left_hand": detection([400, 200, 480, 320])},
            {},
            {"left_hand": detection([420, 200, 500, 320])},
        ]
        M._fill_short_hand_gaps(records, 3)
        self.assertEqual(records[1]["left_hand"]["bbox"], [410, 200, 490, 320])
        self.assertEqual(records[1]["left_hand"]["bbox_extent"], "amodal_estimated")

    def test_ark_anchor_requires_temporal_identity_when_hints_exist(self):
        records = [dict() for _ in range(9)]
        records[2]["left_hand"] = detection([100, 240, 180, 360], 0.8)
        records[6]["left_hand"] = detection([110, 240, 190, 360], 0.8)
        accepted_primary, _ = M._ark_hand_anchor_in_primary_scope(
            records, 4, "left_hand", [105, 240, 185, 360], 0.35, 1000, 600
        )
        accepted_impostor, scores = M._ark_hand_anchor_in_primary_scope(
            records, 4, "left_hand", [460, 240, 540, 360], 0.35, 1000, 600
        )
        self.assertTrue(accepted_primary)
        self.assertFalse(accepted_impostor)
        self.assertEqual(scores["decision_mode"], "temporal_identity_required")

    def test_duplicate_hand_boxes_keep_only_relation_expected_hand(self):
        records = [
            {
                "left_hand": detection([400, 200, 520, 360], 0.99, "sam3_track+primary_instance_lock"),
                "right_hand": detection([402, 202, 522, 362], 0.80, "sam3_track+primary_instance_lock"),
            }
        ]
        specs = [
            {
                "hand_relations": [
                    {"step_id": 1, "hand": "right hand", "relation": "holding"}
                ]
            }
        ]
        steps = {
            1: {
                "step_id": 1,
                "start_frame_0based": 0,
                "end_frame_0based_exclusive": 1,
            }
        }
        meta = M._resolve_duplicate_primary_hand_boxes(records, specs, steps, 30.0)
        self.assertNotIn("left_hand", records[0])
        self.assertIn("right_hand", records[0])
        self.assertEqual(meta["resolved_duplicate_frame_count"], 1)
        self.assertEqual(
            meta["resolution_sample"][0]["decision_mode"],
            "single_relation_expected_hand",
        )

    def test_duplicate_hand_boxes_with_both_expected_keep_stronger_evidence(self):
        records = [
            {
                "left_hand": detection([400, 200, 520, 360], 0.95, "wilor_yolo"),
                "right_hand": detection([402, 202, 522, 362], 0.80, "wilor_yolo"),
            }
        ]
        specs = [
            {
                "hand_relations": [
                    {"step_id": 1, "hand": "both hands", "relation": "holding"}
                ]
            }
        ]
        steps = {
            1: {
                "step_id": 1,
                "start_frame_0based": 0,
                "end_frame_0based_exclusive": 1,
            }
        }
        meta = M._resolve_duplicate_primary_hand_boxes(records, specs, steps, 30.0)
        self.assertIn("left_hand", records[0])
        self.assertNotIn("right_hand", records[0])
        self.assertEqual(
            meta["resolution_sample"][0]["decision_mode"],
            "both_relations_evidence_priority",
        )


class Sam3HandFusionTests(unittest.TestCase):
    def test_locked_instance_geometry_drift_is_rejected_even_when_id_is_unchanged(self):
        records = []
        frames = {}
        for frame_idx in range(7):
            primary = [420 + frame_idx * 3, 250, 520 + frame_idx * 3, 390]
            sam3_box = [780, 0, 930, 170] if frame_idx == 3 else primary
            records.append(
                {} if frame_idx == 3 else {"right_hand": detection(primary, 0.8, "wilor")}
            )
            frames[str(frame_idx)] = [
                {
                    "instance_id": "seg00000_obj1",
                    "bbox_xyxy_pixel": sam3_box,
                    "confidence": 0.9,
                }
            ]
        payload = {"requests": {"__hands__": {"frames": frames}}}

        meta = M._fuse_sam3_hands(records, payload, 0, 1000, 600)

        self.assertNotIn("right_hand", records[3])
        self.assertEqual(meta["selected_instance_transition_rejections"], 1)
        self.assertEqual(records[4]["right_hand"]["sam3_instance_id"], "seg00000_obj1")

    def test_one_primary_instance_per_hand_blocks_low_support_secondary(self):
        records = []
        frames = {}
        for frame_idx in range(8):
            primary = [400 + frame_idx * 3, 260, 500 + frame_idx * 3, 390]
            secondary = [720 + frame_idx * 3, 0, 850 + frame_idx * 3, 170]
            observed = secondary if frame_idx in {3, 4} else primary
            records.append({"right_hand": detection(observed, 0.8, "wilor")})
            frames[str(frame_idx)] = [
                {
                    "instance_id": "seg00000_obj1",
                    "bbox_xyxy_pixel": primary,
                    "confidence": 0.9,
                },
                {
                    "instance_id": "seg00000_obj2",
                    "bbox_xyxy_pixel": secondary,
                    "confidence": 0.95,
                },
            ]
        payload = {"requests": {"__hands__": {"frames": frames}}}

        meta = M._fuse_sam3_hands(records, payload, 0, 1000, 600)

        selected = meta["primary_instance_assignments"]["seg00000"][
            "selected_instance_ids"
        ]
        self.assertEqual(selected["right_hand"], ["seg00000_obj1"])
        self.assertEqual(selected["left_hand"], [])
        for frame_idx, record in enumerate(records):
            self.assertEqual(
                record["right_hand"]["bbox"],
                [400 + frame_idx * 3, 260, 500 + frame_idx * 3, 390],
            )
            self.assertEqual(
                record["right_hand"]["sam3_instance_id"], "seg00000_obj1"
            )
        self.assertEqual(meta["stable_instances_with_handedness"], 1)

    def test_segment_boundary_continuity_inherits_primary_hand_identity(self):
        records = []
        frames = {}
        for frame_idx in range(6):
            primary = [420 + frame_idx * 4, 250, 520 + frame_idx * 4, 390]
            secondary = [780, 0, 920, 160]
            records.append(
                {"right_hand": detection(primary, 0.8, "wilor")}
                if frame_idx < 3
                else {}
            )
            segment = "seg00000" if frame_idx < 3 else "seg00001"
            frames[str(frame_idx)] = [
                {
                    "instance_id": f"{segment}_obj1",
                    "bbox_xyxy_pixel": primary,
                    "confidence": 0.9,
                },
                {
                    "instance_id": f"{segment}_obj2",
                    "bbox_xyxy_pixel": secondary,
                    "confidence": 0.95,
                },
            ]
        payload = {"requests": {"__hands__": {"frames": frames}}}

        meta = M._fuse_sam3_hands(records, payload, 0, 1000, 600)

        assignments = meta["primary_instance_assignments"]
        self.assertEqual(
            assignments["seg00000"]["selected_instance_ids"]["right_hand"],
            ["seg00000_obj1"],
        )
        self.assertEqual(
            assignments["seg00001"]["selected_instance_ids"]["right_hand"],
            ["seg00001_obj1"],
        )
        for frame_idx in range(3, 6):
            self.assertEqual(
                records[frame_idx]["right_hand"]["sam3_instance_id"],
                "seg00001_obj1",
            )

    def test_long_segment_rejects_two_frame_secondary_only_evidence(self):
        records = [dict() for _ in range(500)]
        frames = {}
        for frame_idx in {200, 201}:
            box = [760, 0, 900, 150]
            records[frame_idx]["right_hand"] = detection(box, 0.9, "secondary")
            frames[str(frame_idx)] = [
                {
                    "instance_id": "seg00000_obj9",
                    "bbox_xyxy_pixel": box,
                    "confidence": 0.95,
                }
            ]
        payload = {
            "chunk_size": 500,
            "requests": {
                "__hands__": {"ranges": [[0, 500]], "frames": frames}
            },
        }

        meta = M._fuse_sam3_hands(records, payload, 0, 1000, 600)

        selected = meta["primary_instance_assignments"]["seg00000"][
            "selected_instance_ids"
        ]
        self.assertEqual(selected["right_hand"], [])
        self.assertEqual(
            meta["primary_instance_assignments"]["seg00000"][
                "minimum_matched_frames"
            ],
            5,
        )

    def test_non_overlapping_same_segment_fragment_links_to_existing_hand_chain(self):
        records = [dict() for _ in range(9)]
        frames = {}
        for frame_idx in range(4):
            box = [420 + frame_idx * 4, 250, 520 + frame_idx * 4, 390]
            records[frame_idx]["left_hand"] = detection(box, 0.8, "wilor_left")
            frames[str(frame_idx)] = [
                {
                    "instance_id": "seg00000_obj0",
                    "bbox_xyxy_pixel": box,
                    "confidence": 0.9,
                }
            ]
        for frame_idx in range(6, 9):
            box = [444 + (frame_idx - 6) * 4, 250, 544 + (frame_idx - 6) * 4, 390]
            records[frame_idx]["left_hand"] = detection(box, 0.7, "wilor_left")
            frames[str(frame_idx)] = [
                {
                    "instance_id": "seg00000_obj2",
                    "bbox_xyxy_pixel": box,
                    "confidence": 0.85,
                }
            ]
        payload = {"requests": {"__hands__": {"frames": frames}}}

        meta = M._fuse_sam3_hands(records, payload, 3, 1000, 600)

        selected = meta["primary_instance_assignments"]["seg00000"][
            "selected_instance_ids"
        ]["left_hand"]
        self.assertEqual(selected, ["seg00000_obj0", "seg00000_obj2"])
        self.assertEqual(meta["continuity_linked_fragment_count"], 1)
        self.assertEqual(records[6]["left_hand"]["sam3_instance_id"], "seg00000_obj2")
        self.assertEqual(
            records[4]["left_hand"]["identity_validation"],
            "interpolated_between_continuity_linked_instances",
        )


class ObjectGroundingAndTrackTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "track_id": "object_cup_001",
            "label": "cup",
            "active_intervals": [[0, 10]],
            "active_step_ids": [1],
            "hand_relations": [
                {"step_id": 1, "hand": "right hand", "relation": "grasping"}
            ],
        }
        self.steps = {
            1: {
                "step_id": 1,
                "start_frame_0based": 0,
                "end_frame_0based_exclusive": 10,
            }
        }
        self.hands = [dict() for _ in range(10)]
        self.hands[5]["right_hand"] = detection([440, 230, 540, 370], 0.9)

    def test_api_object_anchor_needs_primary_hand_evidence_without_sam3(self):
        near_anchor = {
            "object_cup_001": {
                "frame_index_0based": 5,
                "bbox_xyxy_pixel": [520, 250, 640, 390],
            }
        }
        accepted, report = M._validate_api_object_anchors(
            [self.spec], near_anchor, {}, self.hands, self.steps, 30.0, 10, 1000, 600
        )
        self.assertIn("object_cup_001", accepted)
        self.assertTrue(report["object_cup_001"]["accepted"])

        edge_anchor = {
            "object_cup_001": {
                "frame_index_0based": 5,
                "bbox_xyxy_pixel": [5, 5, 80, 80],
            }
        }
        accepted, report = M._validate_api_object_anchors(
            [self.spec], edge_anchor, {}, self.hands, self.steps, 30.0, 10, 1000, 600
        )
        self.assertNotIn("object_cup_001", accepted)
        self.assertIn("no_primary_hand_interaction_evidence", report["object_cup_001"]["reason"])

        hand_duplicate = {
            "object_cup_001": {
                "frame_index_0based": 5,
                "bbox_xyxy_pixel": [440, 230, 540, 370],
            }
        }
        accepted, report = M._validate_api_object_anchors(
            [self.spec], hand_duplicate, {}, self.hands, self.steps, 30.0, 10, 1000, 600
        )
        self.assertNotIn("object_cup_001", accepted)
        self.assertIn("duplicates_primary_hand_bbox", report["object_cup_001"]["reason"])

    def test_unknown_hand_relation_uses_both_primary_boxes_for_grounding(self):
        spec = {
            **self.spec,
            "hand_relations": [
                {"step_id": 1, "hand": "unknown hand", "relation": "holding"}
            ],
        }
        self.hands[5]["left_hand"] = detection([300, 230, 400, 370], 0.9)
        boxes = M._relevant_hand_boxes_for_spec(
            spec, 5, self.hands, {1: (0, 10)}
        )
        self.assertEqual(len(boxes), 2)
        expected = M._collect_expected_hand_steps([spec])
        self.assertIn(1, expected["left_hand"])
        self.assertIn(1, expected["right_hand"])

    def test_sam3_object_selection_rejects_hand_duplicate_instance(self):
        payload = {
            "requests": {
                "object_cup_001": {
                    "frames": {
                        "5": [
                            {
                                "instance_id": "segment_000_obj001",
                                "bbox_xyxy_pixel": [440, 230, 540, 370],
                                "confidence": 0.99,
                            },
                            {
                                "instance_id": "segment_000_obj002",
                                "bbox_xyxy_pixel": [520, 250, 640, 390],
                                "confidence": 0.70,
                            },
                        ]
                    }
                }
            }
        }
        raw, anchors, meta = M._select_sam3_object_tracks(
            [self.spec], self.steps, payload, self.hands, 30.0, 10, 1000, 600
        )
        self.assertEqual(raw["object_cup_001"][5]["bbox"], [520, 250, 640, 390])
        self.assertEqual(
            anchors["object_cup_001"]["sam3_instance_id"],
            "segment_000_obj002",
        )
        self.assertEqual(
            meta["object_cup_001"]["selected_instance_ids"],
            ["segment_000_obj002"],
        )

    def test_weak_track_optical_flow_augmentation_is_interval_scoped(self):
        raw = {
            1: {"bbox": [100, 100, 180, 180], "confidence": 0.4, "source": "sam3"},
            6: {"bbox": [300, 100, 380, 180], "confidence": 0.4, "source": "sam3"},
        }
        original = M._optical_flow_track

        def fake_track(_paths, seed_idx, seed_box, _width, _height, start, end):
            return {
                frame_idx: {
                    "bbox": [seed_box[0] + frame_idx - seed_idx, seed_box[1], seed_box[2] + frame_idx - seed_idx, seed_box[3]],
                    "confidence": 0.8,
                    "source": "fake_flow",
                }
                for frame_idx in range(start, end)
            }

        M._optical_flow_track = fake_track
        try:
            augmented, meta = M._augment_track_with_optical_flow(
                raw,
                [Path(f"frame_{index}.jpg") for index in range(10)],
                [[0, 4], [5, 9]],
                {"frame_index_0based": 1, "bbox_xyxy_pixel": [100, 100, 180, 180]},
                1000,
                600,
                0.55,
                "sam3_anchor",
            )
        finally:
            M._optical_flow_track = original
        self.assertEqual(len(meta["seeds"]), 2)
        self.assertTrue(all(frame_idx in augmented for frame_idx in [0, 2, 3, 5, 7, 8]))
        self.assertNotIn(4, augmented)
        self.assertNotIn(9, augmented)

    def test_amodal_track_has_one_record_per_original_frame(self):
        track = M._amodalize_track(
            {
                1: {"bbox": [100, 100, 180, 180], "confidence": 0.8},
                3: {"bbox": [110, 100, 190, 180], "confidence": 0.8},
            },
            [100, 100, 180, 180],
            5,
            1000,
            600,
            [[0, 5]],
        )
        self.assertEqual(len(track), 5)
        self.assertEqual(track[2]["visibility"], "occluded")
        self.assertIsNone(track[2]["bbox"])


class ApiTriggerTests(unittest.TestCase):
    def setUp(self):
        self.specs = [
            {
                "track_id": "object_cup_001",
                "label": "cup",
                "active_intervals": [[0, 20]],
                "active_step_ids": [1],
                "hand_relations": [
                    {"step_id": 1, "hand": "right hand", "relation": "holding"}
                ],
            }
        ]
        self.steps = {
            1: {
                "step_id": 1,
                "step_goal": "Hold the cup.",
                "start_frame_0based": 0,
                "end_frame_0based_exclusive": 20,
            }
        }
        self.paths = [Path(f"frame_{index}.jpg") for index in range(20)]

    def _run_hand_assist(
        self,
        missing_indices,
        return_detections=True,
        confirm_not_visible=False,
        not_visible_reason="out_of_frame",
    ):
        records = [
            ({"right_hand": detection([440, 230, 540, 370], 0.9)} if index not in missing_indices else {})
            for index in range(20)
        ]
        calls = []

        class FakeGrounder:
            def __init__(self, *_args, **_kwargs):
                self.usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_calls": 0,
                    "api_attempts": 0,
                    "cache_hits": 0,
                }
                self.calls_by_step = {}

            def ground(self, _path, labels, _contexts, step_id, _width, _height):
                calls.append((step_id, tuple(labels), ("TARGET",)))
                self.usage["api_calls"] += 1
                self.calls_by_step[step_id] = self.calls_by_step.get(step_id, 0) + 1
                detections = {
                    label: (
                        {
                            "bbox": None,
                            "visibility": "not_visible",
                            "not_visible_reason": not_visible_reason,
                        }
                        if confirm_not_visible
                        else {
                            "bbox": [445, 230, 545, 370],
                            "visibility": "full",
                        }
                    )
                    for label in labels
                }
                return detections if return_detections else {}

            def ground_temporal(
                self,
                _target_path,
                context_frames,
                labels,
                _contexts,
                step_id,
                _width,
                _height,
            ):
                calls.append(
                    (step_id, tuple(labels), tuple(role for role, _path in context_frames))
                )
                self.usage["api_calls"] += 1
                self.calls_by_step[step_id] = self.calls_by_step.get(step_id, 0) + 1
                detections = {
                    label: (
                        {
                            "bbox": None,
                            "visibility": "not_visible",
                            "not_visible_reason": not_visible_reason,
                        }
                        if confirm_not_visible
                        else {
                            "bbox": [445, 230, 545, 370],
                            "visibility": "full",
                        }
                    )
                    for label in labels
                }
                return detections if return_detections else {}

        original_grounder = M.ArkHandGrounder
        original_flow = M._optical_flow_track
        M.ArkHandGrounder = FakeGrounder
        M._optical_flow_track = lambda _paths, frame_idx, box, *_args: {
            frame_idx: {"bbox": box, "confidence": 1.0, "source": "test_anchor"}
        }
        try:
            cfg = M.BBoxConfig(
                api_hand_assist=True,
                api_hand_min_missing_run_frames=4,
                api_hand_low_conf_threshold=0.35,
                api_hand_max_calls_per_step=3,
                api_hand_max_total_calls=24,
            )
            meta = M._assist_hands_with_ark(
                self.specs,
                self.steps,
                self.paths,
                records,
                30.0,
                1000,
                600,
                cfg,
                types.SimpleNamespace(),
                Path("unused"),
            )
        finally:
            M.ArkHandGrounder = original_grounder
            M._optical_flow_track = original_flow
        return calls, meta

    def test_hand_api_not_triggered_for_three_frame_gap(self):
        calls, meta = self._run_hand_assist({5, 6, 7})
        self.assertEqual(calls, [])
        self.assertFalse(meta["triggered"])

    def test_hand_api_triggered_for_four_frame_gap(self):
        calls, meta = self._run_hand_assist({5, 6, 7, 8})
        self.assertEqual(len(calls), 1)
        self.assertTrue(meta["triggered"])
        self.assertEqual(meta["trigger_runs"][0]["length_frames"], 4)
        self.assertTrue(any(role.startswith("PREVIOUS") for role in calls[0][2]))
        self.assertTrue(any(role.startswith("NEXT") for role in calls[0][2]))
        self.assertEqual(
            meta["scheduled_requests"][0]["context_frame_indices_0based"], [4, 9]
        )

    def test_hand_api_not_visible_result_is_audited_as_rejected(self):
        calls, meta = self._run_hand_assist({5, 6, 7, 8}, return_detections=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(meta["rejected_anchors"]), 1)
        self.assertEqual(
            meta["rejected_anchors"][0]["reason"],
            "api_returned_not_visible_or_no_valid_bbox",
        )

    def test_hand_api_structured_not_visible_confirms_interval_without_rejection(self):
        calls, meta = self._run_hand_assist(
            {5, 6, 7, 8}, confirm_not_visible=True
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(meta["rejected_anchors"], [])
        self.assertEqual(len(meta["confirmed_not_visible_intervals"]), 1)
        self.assertEqual(meta["confirmed_not_visible_frames"], 4)
        self.assertEqual(
            meta["confirmed_not_visible_intervals"][0]["output_visibility"],
            "not_present",
        )

    def test_hand_api_unknown_not_visible_reason_does_not_confirm_interval(self):
        calls, meta = self._run_hand_assist(
            {5, 6, 7, 8},
            confirm_not_visible=True,
            not_visible_reason="unknown",
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(meta["confirmed_not_visible_intervals"], [])
        self.assertEqual(meta["confirmed_not_visible_frames"], 0)
        self.assertEqual(
            meta["rejected_anchors"][0]["reason"],
            "api_not_visible_reason_not_specific",
        )

    def test_hand_api_not_visible_confirmation_windows_are_time_bounded(self):
        windows = M._run_sample_windows(0, 90, 30.0)
        self.assertEqual(len(windows), 3)
        self.assertTrue(all(end - start <= 15 for _frame, start, end in windows))
        self.assertEqual(windows[0][1], 0)
        self.assertEqual(windows[-1][2], 90)
        self.assertLess(sum(end - start for _frame, start, end in windows), 90)

    def test_short_hand_gap_uses_one_complete_confirmation_window(self):
        self.assertEqual(M._run_sample_windows(10, 23, 30.0), [(16, 10, 23)])

    def test_object_quality_thresholds_trigger_ark_assist(self):
        cfg = M.BBoxConfig(
            api_object_min_track_coverage=0.60,
            api_object_min_mean_confidence=0.55,
        )
        raw = {
            "object_cup_001": {
                index: {"bbox": [100, 100, 180, 180], "confidence": 0.54}
                for index in range(6)
            }
        }
        weak, report = M._sam3_object_quality(self.specs, raw, cfg)
        self.assertEqual(len(weak), 1)
        self.assertIn(
            "low_mean_confidence",
            report["object_cup_001"]["ark_assist_reasons"],
        )


class ArkGrounderProtocolTests(unittest.TestCase):
    def test_temporal_multimage_call_preserves_structured_not_visible_result(self):
        captured = {}

        class Completions:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content=(
                                    '[{"target_id":"target_001","label":"right_hand",'
                                    '"visibility":"not_visible",'
                                    '"not_visible_reason":"out_of_frame","bbox":null}]'
                                )
                            )
                        )
                    ],
                    usage=types.SimpleNamespace(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for name in ("previous.jpg", "target.jpg", "next.jpg"):
                path = Path(temp_dir) / name
                path.write_bytes(b"test-jpeg-bytes-" + name.encode("ascii"))
                paths.append(path)
            grounder = object.__new__(M.ArkHandGrounder)
            grounder.client = types.SimpleNamespace(
                chat=types.SimpleNamespace(completions=Completions())
            )
            grounder.model = "test-model"
            grounder.max_tokens = 256
            grounder.thinking_type = "enabled"
            grounder.reasoning_effort = "medium"
            grounder.cache_dir = Path(temp_dir) / "cache"
            grounder.cache_dir.mkdir()
            grounder.max_calls_per_step = 3
            grounder.max_total_calls = 3
            grounder.attempts = 0
            grounder.calls_by_step = Counter()
            grounder.usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_calls": 0,
                "api_attempts": 0,
                "cache_hits": 0,
            }

            result = grounder.ground_temporal(
                paths[1],
                [("PREVIOUS frame 1", paths[0]), ("NEXT frame 3", paths[2])],
                ["right_hand"],
                [{}],
                1,
                1000,
                600,
            )

        self.assertEqual(result["right_hand"]["visibility"], "not_visible")
        self.assertEqual(
            result["right_hand"]["not_visible_reason"], "out_of_frame"
        )
        content = captured["messages"][0]["content"]
        self.assertEqual(
            sum(item.get("type") == "image_url" for item in content), 3
        )
        self.assertEqual(captured["reasoning_effort"], "medium")
        self.assertEqual(captured["extra_body"], {"thinking": {"type": "enabled"}})


class QualityReportTests(unittest.TestCase):
    def test_schema_version_invalidates_pre_audit_stage5_outputs(self):
        self.assertEqual(
            M.STAGE5_SCHEMA_VERSION,
            "stage5-bbox-v9-bounded-visibility-evidence",
        )

    def test_warning_stage5_output_is_never_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            jsonl = root / "frame_bboxes.jsonl"
            jsonl.write_text("{}\n", encoding="utf-8")
            cfg = M.BBoxConfig()
            stat = video.stat()
            summary = {
                "schema_version": M.STAGE5_SCHEMA_VERSION,
                "source_video": {"path": str(video)},
                "source_video_stat": {
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                },
                "final_plan_sha256": "plan-hash",
                "bbox_config": M.asdict(cfg),
                "total_frames": 1,
                "quality_gate": {
                    "status": "warning",
                    "eligible_for_automatic_use": False,
                },
            }
            summary_path = root / "stage5_summary.json"
            summary_path.write_text(M.json.dumps(summary), encoding="utf-8")
            self.assertFalse(
                M._can_resume_stage5(
                    summary_path,
                    jsonl,
                    str(video),
                    1,
                    "plan-hash",
                    cfg,
                )
            )
            summary["quality_gate"] = {
                "status": "passed",
                "eligible_for_automatic_use": True,
            }
            summary_path.write_text(M.json.dumps(summary), encoding="utf-8")
            self.assertTrue(
                M._can_resume_stage5(
                    summary_path,
                    jsonl,
                    str(video),
                    1,
                    "plan-hash",
                    cfg,
                )
            )

    def test_quality_report_flags_realistic_cross_person_hand_switch(self):
        boxes = [
            [1191, 493, 1346, 624],
            [1287, 0, 1365, 235],
            [1304, 0, 1387, 252],
            [1305, 564, 1433, 704],
        ]
        records = []
        for frame_idx, box in enumerate(boxes):
            records.append(
                {
                    "frame_index_0based": frame_idx,
                    "entities": [
                        {
                            "track_id": "hand_right_001",
                            "entity_type": "hand",
                            "label": "right_hand",
                            "bbox_xyxy_pixel": box,
                            "bbox_extent": "observed_full",
                            "confidence": 0.9,
                            "visibility": "visible",
                            "source": "sam3",
                            "is_active_manipulation": True,
                            "sam3_instance_id": (
                                "seg00000_obj1" if frame_idx in {0, 3} else "seg00000_obj2"
                            ),
                        }
                    ],
                }
            )
        report, warnings = M._track_quality_report(records, 1920, 1080)
        quality = report["hand_right_001"]
        self.assertGreaterEqual(quality["temporal_identity_break_count"], 2)
        self.assertGreaterEqual(quality["sam3_instance_switch_count"], 2)
        self.assertTrue(
            any(item["reason"] == "hand_temporal_identity_discontinuity" for item in warnings)
        )

    def test_quality_report_does_not_call_same_chain_border_reentry_identity_break(self):
        records = []
        for frame_idx, box in enumerate(([990, 0, 999, 20], [850, 0, 980, 180])):
            records.append(
                {
                    "frame_index_0based": frame_idx,
                    "entities": [
                        {
                            "track_id": "hand_left_001",
                            "entity_type": "hand",
                            "label": "left_hand",
                            "bbox_xyxy_pixel": box,
                            "bbox_extent": "observed_full",
                            "confidence": 0.9,
                            "visibility": "visible",
                            "source": "sam3",
                            "is_active_manipulation": True,
                            "sam3_instance_id": "seg00000_obj0",
                            "primary_hand_chain_id": "primary_operator_001:left_hand",
                        }
                    ],
                }
            )
        report, warnings = M._track_quality_report(records, 1000, 600)
        quality = report["hand_left_001"]
        self.assertEqual(quality["temporal_identity_break_count"], 0)
        self.assertEqual(quality["abrupt_area_change_count"], 0)
        self.assertEqual(quality["border_clipped_area_change_count"], 1)
        self.assertFalse(
            any(item["reason"] == "hand_temporal_identity_discontinuity" for item in warnings)
        )

    def test_api_confirmed_not_visible_gap_is_not_unexplained_missing(self):
        records = []
        for frame_idx in range(5):
            records.append(
                {
                    "frame_index_0based": frame_idx,
                    "entities": [
                        {
                            "track_id": "hand_right_001",
                            "entity_type": "hand",
                            "label": "right_hand",
                            "bbox_xyxy_pixel": None,
                            "bbox_extent": None,
                            "confidence": 1.0,
                            "visibility": "not_present",
                            "source": "ark_hand_visibility_confirmation",
                            "not_visible_reason": "out_of_frame",
                            "is_active_manipulation": True,
                        }
                    ],
                }
            )
        report, _warnings = M._track_quality_report(records, 1000, 600)
        quality = report["hand_right_001"]
        self.assertEqual(quality["longest_consecutive_missing_frames"], 5)
        self.assertEqual(quality["longest_unexplained_missing_frames"], 0)

    def test_unknown_api_visibility_reason_remains_unexplained(self):
        records = []
        for frame_idx in range(5):
            records.append(
                {
                    "frame_index_0based": frame_idx,
                    "entities": [
                        {
                            "track_id": "hand_right_001",
                            "entity_type": "hand",
                            "label": "right_hand",
                            "bbox_xyxy_pixel": None,
                            "bbox_extent": None,
                            "confidence": 1.0,
                            "visibility": "not_visible",
                            "source": "ark_hand_visibility_confirmation",
                            "not_visible_reason": "unknown",
                            "is_active_manipulation": True,
                        }
                    ],
                }
            )
        report, _warnings = M._track_quality_report(records, 1000, 600)
        quality = report["hand_right_001"]
        self.assertEqual(quality["longest_unexplained_missing_frames"], 5)

    def test_quality_report_flags_sparse_active_object_track(self):
        records = []
        for frame_idx in range(5):
            object_box = [100, 100, 180, 180] if frame_idx in {0, 4} else None
            records.append(
                {
                    "frame_index_0based": frame_idx,
                    "entities": [
                        {
                            "track_id": "object_cup_001",
                            "entity_type": "object",
                            "label": "cup",
                            "bbox_xyxy_pixel": object_box,
                            "bbox_extent": "observed_full" if object_box else None,
                            "confidence": 0.8 if object_box else 0.0,
                            "visibility": "visible" if object_box else "occluded",
                            "source": "sam3" if object_box else "none",
                            "is_active_manipulation": True,
                        }
                    ],
                }
            )
        report, warnings = M._track_quality_report(records, 1000, 600)
        self.assertEqual(report["object_cup_001"]["bbox_coverage"], 0.4)
        self.assertEqual(report["object_cup_001"]["longest_consecutive_missing_frames"], 3)
        self.assertTrue(any("coverage_below_0.60" in item["reason"] for item in warnings))


class RuntimeIsolationTests(unittest.TestCase):
    def test_cpu_hand_detector_cannot_hide_cuda_from_sam3(self):
        name = "CUDA_VISIBLE_DEVICES"
        existed = name in M.os.environ
        original = M.os.environ.get(name)
        try:
            M.os.environ[name] = "0"
            with M._preserve_environment_variable(name):
                M.os.environ[name] = "-1"
            self.assertEqual(M.os.environ.get(name), "0")

            M.os.environ.pop(name, None)
            with M._preserve_environment_variable(name):
                M.os.environ[name] = "-1"
            self.assertNotIn(name, M.os.environ)
        finally:
            if existed:
                M.os.environ[name] = str(original)
            else:
                M.os.environ.pop(name, None)


if __name__ == "__main__":
    unittest.main()
