from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_stage2h():
    common = types.ModuleType("common")
    common.ApiConfig = object
    for name in (
        "build_video_api_content", "call_chat_completion", "ensure_video_out_dir_safe",
        "extract_json_from_response", "format_duration", "guard_schema_fingerprint",
        "initialize_api_client", "now_utc_iso", "read_json", "update_run_summary",
        "video_id_from_path", "write_json",
    ):
        setattr(common, name, lambda *args, **kwargs: None)
    common.logger = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)

    stage5 = types.ModuleType("stage5_bbox_tracking")
    stage5.BBoxConfig = object
    stage5.default_hand_detector_path = lambda: "detector.pt"
    stage5._resolve_hand_device = lambda value: value

    spec = importlib.util.spec_from_file_location("stage2h_value_gate_target", ROOT / "stage2h_qc_hand_depth.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"common": common, "stage5_bbox_tracking": stage5, spec.name: module}):
        spec.loader.exec_module(module)
    return module


M = _load_stage2h()


def detected(score=0.9):
    return [{"score": score, "bbox_xyxy": [0, 0, 10, 10], "hand": "right"}]


class ContinuousHandGateTests(unittest.TestCase):
    def run_length(self, positive_frames):
        hand_frames = {frame: detected() for frame in positive_frames}
        return M._longest_continuous_hand_run(
            start_frame=0, end_frame=100, frame_stride=10,
            hand_frames=hand_frames, hand_score_threshold=0.5, source_fps=30.0,
        )

    def test_exactly_one_second_is_not_enough(self):
        count, seconds = self.run_length([0, 10, 20, 30])
        self.assertEqual(count, 4)
        self.assertAlmostEqual(seconds, 1.0)
        self.assertEqual("yes" if seconds > 1.0 else "no", "no")

    def test_strictly_more_than_one_second_passes(self):
        count, seconds = self.run_length([0, 10, 20, 30, 40])
        self.assertEqual(count, 5)
        self.assertGreater(seconds, 1.0)

    def test_gap_breaks_the_continuous_run(self):
        count, seconds = self.run_length([0, 10, 20, 40, 50, 60])
        self.assertEqual(count, 3)
        self.assertAlmostEqual(seconds, 2 / 3)


class VlmResponseTests(unittest.TestCase):
    def test_accepts_explainable_yes_no(self):
        value, reason, errors = M._validate_vlm_action_value({"value": "yes", "reason": "The hand pours liquid."})
        self.assertEqual(value, "yes")
        self.assertTrue(reason)
        self.assertEqual(errors, [])

    def test_rejects_missing_reason(self):
        value, _, errors = M._validate_vlm_action_value({"value": "yes"})
        self.assertIsNone(value)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
