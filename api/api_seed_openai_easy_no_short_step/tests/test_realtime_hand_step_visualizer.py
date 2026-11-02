import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import realtime_hand_step_visualizer as visualizer


class RealtimeHandStepVisualizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "hand_steps": [
                {
                    "hand_step_id": 1,
                    "start_frame_id": 10,
                    "end_frame_id": 20,
                    "active_hands": ["left_hand", "right_hand"],
                    "action_semantics": {
                        "left_hand": "The left hand steadies the cup.",
                        "right_hand": "The right hand lifts the cup.",
                        "overall": "Both hands lift and stabilize the cup.",
                    },
                    "detection_sources": ["vlm", "upstream"],
                    "hand_object_relations": [
                        {"hand": "left hand", "relation": "holding", "object": "cup", "entity_ref": "object_001"}
                    ],
                    "vlm_frame_annotations": [
                        {
                            "frame_id": 12,
                            "has_effective_hand_action": 1,
                            "active_hands": ["left_hand", "right_hand"],
                            "confidence": 0.91,
                            "action_semantics": {
                                "left_hand": "The left hand steadies the cup.",
                                "right_hand": "The right hand lifts the cup.",
                                "overall": "Both hands manipulate the cup.",
                            },
                        }
                    ],
                    "visibility": {
                        "left_hand": {"arm_visible": 1, "hand_completeness_score": 0.75},
                        "right_hand": {"arm_visible": 0, "hand_completeness_score": 0.5},
                        "overall_hand_completeness_score": 0.5,
                        "arm_visible_any": 1,
                        "aggregation": "minimum_over_participating_hands",
                    },
                    "frame_visibility": [
                        {
                            "frame_id": 12,
                            "left_hand": {"arm_visible": 1, "hand_completeness_score": 0.75},
                            "right_hand": {"arm_visible": 0, "hand_completeness_score": 0.5},
                        }
                    ],
                    "bbox_evaluations": [
                        {"frame_id": 12, "hand": "left_hand", "bbox_source": "upstream_sam3", "bbox_xyxy": [10, 10, 30, 30], "arm_visible": 1, "hand_completeness_score": 0.72},
                        {"frame_id": 12, "hand": "right_hand", "bbox_source": "vlm_generated", "bbox_xyxy": [40, 10, 60, 30], "arm_visible": 0, "hand_completeness_score": 0.44},
                    ],
                    "bbox_tracks": {
                        "left_hand": [
                            {
                                "frame_id": 12,
                                "bbox_xyxy": [10, 10, 30, 30],
                                "bbox_source": "upstream",
                            }
                        ],
                        "right_hand": [
                            {
                                "frame_id": 12,
                                "bbox_xyxy": [40, 10, 60, 30],
                                "bbox_source": "vlm_grounding",
                            }
                        ],
                    },
                }
            ]
        }

    def test_same_frame_indexes_sam3_and_vlm_boxes(self) -> None:
        index = visualizer.HandStepIndex(self.payload)
        boxes = list(index.boxes_for_frame(12))
        self.assertEqual(len(boxes), 2)
        self.assertEqual({box.source for box in boxes}, {"SAM3_bbox", "VLM_generated_bbox"})
        self.assertEqual(next(box for box in boxes if box.source == "SAM3_bbox").hand_completeness_score, 0.72)
        self.assertEqual(next(box for box in boxes if box.source == "VLM_generated_bbox").arm_visible, 0)
        self.assertEqual(index.step_for_frame(12)["hand_step_id"], 1)
        self.assertIsNone(index.step_for_frame(21))

    def test_render_appends_semantic_panel_without_writing_files(self) -> None:
        index = visualizer.HandStepIndex(self.payload)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            before = set(os.listdir(tmpdir))
            rendered = visualizer.render_frame(
                frame,
                frame_id=12,
                index=index,
                max_video_width=100,
                panel_width=360,
            )
            after = set(os.listdir(tmpdir))
        self.assertEqual(rendered.shape, (80, 460, 3))
        self.assertEqual(before, after)
        self.assertTrue(np.any(rendered[:, 100:, :]))
        self.assertTrue(np.array_equal(rendered[30, 10], np.array(visualizer.SAM3_COLOR)))
        self.assertTrue(np.array_equal(rendered[30, 40], np.array(visualizer.VLM_COLOR)))

    def test_frame_details_exposes_all_current_frame_products(self) -> None:
        details = visualizer.HandStepIndex(self.payload).details_for_frame(12)
        self.assertTrue(details["inside_hand_step"])
        self.assertEqual(details["vlm_frame_annotation"]["has_effective_hand_action"], 1)
        self.assertEqual(len(details["bbox_evaluations"]), 2)
        self.assertEqual(details["step"]["hand_object_relations"][0]["entity_ref"], "object_001")
        self.assertEqual(details["step"]["bbox_track_counts"], {"SAM3_bbox": 1, "VLM_generated_bbox": 1})

    def test_pipeline_artifacts_include_global_products_and_trim_frame_bulk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage1 = os.path.join(tmpdir, "stage1")
            stage3 = os.path.join(tmpdir, "stage3")
            os.makedirs(stage1)
            os.makedirs(stage3)
            hand_steps_path = os.path.join(stage3, "hand_steps.json")
            with open(hand_steps_path, "w", encoding="utf-8") as handle:
                handle.write('{"hand_steps":[]}')
            with open(os.path.join(tmpdir, "final_plan.json"), "w", encoding="utf-8") as handle:
                handle.write('{"steps":[{"step_id":1}]}')
            with open(os.path.join(stage1, "vlm_hand_events.json"), "w", encoding="utf-8") as handle:
                handle.write('{"frame_annotations":[{"frame_id":0}],"vlm_action_frame_ids":[0]}')
            artifacts = visualizer.load_pipeline_artifacts(hand_steps_path)
        self.assertTrue(artifacts["final_plan"]["available"])
        self.assertEqual(artifacts["vlm_hand_frames"]["data"]["frame_annotation_count"], 1)
        self.assertNotIn("frame_annotations", artifacts["vlm_hand_frames"]["data"])

    def test_visibility_summary_uses_only_exact_representative_frame(self) -> None:
        step = self.payload["hand_steps"][0]
        self.assertEqual(visualizer.visibility_summary(step, "left_hand", 12), "arm: yes | exposure: 0.75")
        self.assertEqual(visualizer.visibility_summary(step, "right_hand", 12), "arm: no | exposure: 0.50")
        self.assertEqual(visualizer.visibility_summary(step, "overall", 12), "arm any: yes | exposure: 0.50")
        self.assertEqual(
            visualizer.visibility_summary(step, "left_hand", 13),
            "arm: N/A | exposure: N/A | representative frame: no",
        )

    def test_browser_page_has_pause_seek_and_single_frame_controls(self) -> None:
        page = visualizer._html_page("/data/raw_data/rgb.mp4", "/data/stage3/hand_steps.json").decode("utf-8")
        self.assertIn('id="play"', page)
        self.assertIn('id="progress"', page)
        self.assertIn('id="prev"', page)
        self.assertIn('id="next"', page)
        self.assertIn("/frame.jpg?frame_id=", page)
        self.assertIn("/metadata.json", page)
        self.assertIn("/annotation.json?frame_id=", page)
        self.assertIn("Frame-level VLM decision", page)
        self.assertIn("Bboxes and source-specific quality", page)
        self.assertIn("Hand-object relations", page)
        self.assertIn("Raw current-frame JSON", page)
        self.assertIn("Pipeline JSON artifacts", page)
        self.assertIn("/artifacts.json", page)
        self.assertIn('<option value="256">256x</option>', page)
        self.assertIn("frameAdvance", page)


if __name__ == "__main__":
    unittest.main()
