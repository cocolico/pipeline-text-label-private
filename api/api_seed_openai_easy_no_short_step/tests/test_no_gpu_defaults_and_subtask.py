from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# The committed test lives in ``api/.../tests``; this fallback makes the same
# test runnable from the local staging directory before deployment.
if not (ROOT / "pipeline.py").exists():
    ROOT = Path(__file__).resolve().parent


def _load_stage3():
    # Stage 3 imports are CPU-safe. Use real dependencies rather than replacing
    # process-wide modules and contaminating the rest of unittest discovery.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("stage3_subtask_target", ROOT / "stage3_refine_and_keyframes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoGpuDefaultTests(unittest.TestCase):
    def test_pipeline_default_is_cpu_safe_stages_only(self):
        tree = ast.parse((ROOT / "pipeline.py").read_text(encoding="utf-8"))
        stages_default = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--stages":
                    stages_default = [kw.value.value for kw in node.keywords if kw.arg == "default"]
        self.assertEqual(stages_default, ["1,2,3,4"])

    def test_short_video_threshold_is_fifteen_seconds(self):
        tree = ast.parse((ROOT / "stage2_localize_and_cut.py").read_text(encoding="utf-8"))
        values = [
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "SHORT_VIDEO_SINGLE_STEP_THRESHOLD_SEC"
        ]
        self.assertEqual(values, [15.0])


class SubtaskHandEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_stage3()

    def test_preserves_left_and_right_hand_descriptions(self):
        plan = {"high_level_goal": "prepare drink", "steps": [{
            "step_goal": "The right hand lifts the cup while the left hand stabilizes the table.",
            "hand_object_relations": [
                {"hand": "right hand", "relation": "lifting", "object": "cup"},
                {"hand": "left hand", "relation": "stabilizing", "object": "table"},
            ],
        }]}
        with tempfile.TemporaryDirectory() as tmp:
            old_write = self.module.write_json
            try:
                self.module.write_json = lambda path, value: Path(path).write_text(json.dumps(value), encoding="utf-8")
                self.module._write_instruction_subtask(plan, tmp)
            finally:
                self.module.write_json = old_write
            text = json.loads((Path(tmp) / "instruction_subtask.json").read_text())["sub_task"][0]
        self.assertIn("right hand lifting cup", text)
        self.assertIn("left hand stabilizing table", text)

    def test_records_unknown_laterality_explicitly(self):
        plan = {"high_level_goal": "prepare drink", "steps": [{
            "step_goal": "A hand wipes the counter.",
            "hand_object_relations": [{"hand": "unknown hand", "relation": "wiping", "object": "counter"}],
        }]}
        with tempfile.TemporaryDirectory() as tmp:
            old_write = self.module.write_json
            try:
                self.module.write_json = lambda path, value: Path(path).write_text(json.dumps(value), encoding="utf-8")
                self.module._write_instruction_subtask(plan, tmp)
            finally:
                self.module.write_json = old_write
            text = json.loads((Path(tmp) / "instruction_subtask.json").read_text())["sub_task"][0]
        self.assertIn("left/right not visually distinguishable", text)

    def test_instruction_is_video_level_and_subtasks_are_step_specific(self):
        plan = {
            "high_level_goal": "Prepare the drink and place it on the serving tray.",
            "steps": [
                {
                    "step_goal": "Pick up the metal skewer from the work surface.",
                    "hand_object_relations": [{"hand": "right hand", "relation": "grasping", "object": "metal skewer"}],
                },
                {
                    "step_goal": "Place the food piece onto the target plate.",
                    "hand_object_relations": [{"hand": "left hand", "relation": "holding", "object": "food piece"}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            old_write = self.module.write_json
            try:
                self.module.write_json = lambda path, value: Path(path).write_text(json.dumps(value), encoding="utf-8")
                self.module._write_instruction_subtask(plan, tmp)
            finally:
                self.module.write_json = old_write
            payload = json.loads((Path(tmp) / "instruction_subtask.json").read_text())
        self.assertEqual(payload["instruction"], "Pick up the metal skewer from the work surface. Place the food piece onto the target plate.")
        self.assertEqual(len(payload["sub_task"]), 2)
        self.assertIn("right hand grasping metal skewer", payload["sub_task"][0])
        self.assertIn("left hand holding food piece", payload["sub_task"][1])
        self.assertNotEqual(payload["sub_task"][0], payload["sub_task"][1])


if __name__ == "__main__":
    unittest.main()
