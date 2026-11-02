import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_demo import export, validate_response


def response():
    return {'high_level_goal': 'Move the tomato.', 'steps': [{
        'step_id': 1, 'step_goal': 'Move the tomato.', 'start_sec': 0.5,
        'end_sec': 1.5, 'evidence_status': 'uncertain',
        'action_semantics': {'left_hand': 'Laterality uncertain.',
                             'right_hand': 'Laterality uncertain.', 'overall': 'Move the tomato.'},
        'object_inventory': [{'entity_ref': 'object_1', 'canonical_name': 'tomato'}],
        'hand_object_relations': [{'hand': 'unknown_hand', 'object': 'tomato',
                                  'relation': 'holding', 'entity_ref': 'object_1'}]}]}


class DemoTests(unittest.TestCase):
    def test_unknown_laterality_is_valid(self):
        validate_response(response(), 2)

    def test_invalid_timestamps_rejected(self):
        for start, end in [(False, 1), (0, float('nan')), (-1, 1), (1, 1), (0, 3)]:
            obj = response(); obj['steps'][0].update(start_sec=start, end_sec=end)
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_response(obj, 2)

    def test_overlap_rejected(self):
        obj = response(); second = copy.deepcopy(obj['steps'][0]); second['step_id'] = 2
        obj['steps'].append(second)
        with self.assertRaises(ValueError): validate_response(obj, 2)

    def test_unresolved_object_rejected(self):
        obj = response(); obj['steps'][0]['hand_object_relations'][0]['entity_ref'] = 'missing'
        with self.assertRaises(ValueError): validate_response(obj, 2)

    def test_export_restores_nonzero_source_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            export(response(), {'video': {'fps': 30}, 'start_frame': 900, 'end_frame': 960}, run)
            obj = json.loads((run / 'final_plan.json').read_text()); step = obj['steps'][0]
            self.assertEqual(step['start_frame_0based'], 915)
            self.assertEqual(step['end_frame_0based_exclusive'], 945)
            self.assertEqual(step['end_frame_0based_inclusive'], 944)
            self.assertEqual(step['start_sec'], 30.5)
            self.assertFalse(obj['capabilities']['frame_accurate_boundaries'])
            self.assertEqual(obj['review_status'], 'needs_human_review')

    def test_subframe_steps_are_not_silently_expanded(self):
        obj = response(); obj['steps'][0].update(start_sec=0.001, end_sec=0.002)
        validate_response(obj, 2)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            export(obj, {'video': {'fps': 30}, 'start_frame': 0, 'end_frame': 60}, Path(directory))


if __name__ == '__main__':
    unittest.main()
