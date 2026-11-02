import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT.parent/'stage3_refine_and_keyframes.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0,str(ROOT))
from text_annotation_goal import compose_video_goal, first_subtask_sentence

class SentenceTests(unittest.TestCase):
    def test_english_hand_details_and_two_sentences(self):
        self.assertEqual(first_subtask_sentence('Pick up a cup. Move it left; right hand holding cup'), 'Pick up a cup.')
        self.assertEqual(first_subtask_sentence('Pick up a cup; right hand holding cup'), 'Pick up a cup.')
    def test_chinese_and_terminal_punctuation(self):
        self.assertEqual(compose_video_goal(['拿起杯子；右手抓握杯子','倒水。放下杯子；右手扶杯']), '拿起杯子。 倒水。')
        self.assertEqual(first_subtask_sentence('Lift it!; both hands holding'), 'Lift it!')
    def test_decimal_abbreviation_and_order_with_repeated_actions(self):
        self.assertEqual(first_subtask_sentence('Weigh 1.5 kg on the No. 2 scale; left hand holds bowl'), 'Weigh 1.5 kg on the No. 2 scale.')
        self.assertEqual(compose_video_goal(['Wipe the counter','Put down the cloth','Wipe the counter']), 'Wipe the counter. Put down the cloth. Wipe the counter.')
    def test_missing_subtask_fails_instead_of_silently_omitting_a_segment(self):
        for value in ['',None,'; hand holds cup']:
            with self.assertRaises(ValueError): first_subtask_sentence(value)
        self.assertEqual(compose_video_goal([]),'')

class ExportTests(unittest.TestCase):
    def test_final_step_plan_instruction_are_consistent_and_steps_unchanged(self):
        import stage3_refine_and_keyframes as stage3
        steps=[{'step_id':i+1,'step_goal':goal,'hand_object_relations':[],
                'object_inventory':[{'entity_ref':f'object_{i+1}','name':'cup'}]}
               for i,goal in enumerate(['Pick up the cup. Ignore this second sentence.','Place the cup on the tray'])]
        plan={'source_video':{},'high_level_goal':'Primary operator hand actions in upstream keep=true intervals.','steps':copy.deepcopy(steps)}
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'stage3';directory.mkdir()
            stage3._generate_final_plan_without_stage4(temp,plan,str(directory))
            final=json.loads((Path(temp)/'final_plan.json').read_text())
            cached=json.loads((directory/'step_plan.json').read_text())
            legacy=json.loads((directory/'instruction_subtask.json').read_text())
        expected='Pick up the cup. Place the cup on the tray.'
        self.assertEqual(final['high_level_goal'],expected)
        self.assertEqual(cached['high_level_goal'],expected)
        self.assertEqual(legacy['instruction'],expected)
        self.assertEqual(final['steps'],steps)
        self.assertEqual(legacy['sub_task'],[goal['step_goal']+'; no reliable left/right hand-object action is visible' for goal in steps])
    def test_empty_final_plan_marks_missing_subtasks_without_a_business_goal(self):
        import stage3_refine_and_keyframes as stage3
        plan={'source_video':{},'high_level_goal':'Primary operator hand actions in upstream keep=true intervals.','steps':[]}
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'stage3';directory.mkdir()
            stage3._generate_final_plan_without_stage4(temp,plan,str(directory))
            final=json.loads((Path(temp)/'final_plan.json').read_text())
            legacy=json.loads((directory/'instruction_subtask.json').read_text())
        self.assertEqual(final['high_level_goal'],'No valid subtask was produced for this video.')
        self.assertEqual(legacy['instruction'],final['high_level_goal'])
        self.assertEqual(legacy['sub_task'],[])
    def test_active_hand_step_path_uses_specific_goals_without_model_calls(self):
        from unittest.mock import patch
        import hand_step_pipeline as hand
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'demo';(out/'stage2').mkdir(parents=True)
            goals=['Collect the vegetables','Weigh 1.5 kg of potatoes. This sentence is excluded.']
            boundaries=[]
            for i,goal in enumerate(goals,1):
                boundaries.append({'hand_step_id':i,'start_frame_id':i*10,'end_frame_id':i*10+3,
                                   'draft_action_semantics':{'overall':goal}})
                cache=out/'stage3'/f'step_{i:03d}';cache.mkdir(parents=True)
                (cache/'bbox_arm_visibility.json').write_text(json.dumps({'step_id':i,'bbox_arm_visibility':[]}))
            (out/'stage2/hand_step_boundaries.json').write_text(json.dumps({'hand_steps':boundaries}))
            with patch.object(hand,'ensure_video_out_dir_safe'), patch.object(hand,'get_video_metadata',return_value={'fps':30,'total_frames':90}), patch.object(hand,'build_source_video_meta',return_value={}), patch.object(hand,'_require_stage1_step_clip',return_value=str(out/'clip.mp4')), patch.object(hand,'_step_visualization_path',return_value=str(out/'preview.mp4')), patch.object(hand,'_normalize_bbox_arm_intervals',return_value=([],[])), patch.object(hand,'initialize_api_client') as client:
                _,plan,usage=hand.refine_hand_steps_text_only(video_path='demo.mp4',output_root=temp,api_cfg=None,overwrite=False,max_retries=1)
            self.assertEqual(plan['high_level_goal'],'Collect the vegetables. Weigh 1.5 kg of potatoes.')
            self.assertEqual([s['step_goal'] for s in plan['steps']],goals)
            self.assertEqual(usage['api_calls'],0)
            client.assert_not_called()

if __name__=='__main__': unittest.main()
