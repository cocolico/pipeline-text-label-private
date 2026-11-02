#!/usr/bin/env python3
"""Read-only LeRobot source -> local semantic demo; no screening/GPU required."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1] / 'api/api_seed_openai_easy_no_short_step'
HANDS = {'left_hand', 'right_hand', 'both_hands', 'unknown_hand'}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def probe(path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames',
        '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)['streams']
    if len(streams) != 1:
        raise ValueError('Exactly one selected video stream is required')
    stream = streams[0]
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    return {'width': stream['width'], 'height': stream['height'],
            'fps': numerator / denominator, 'frames': int(stream['nb_read_frames'])}


def validate_response(payload, duration):
    def text(value):
        return isinstance(value, str) and bool(value.strip())

    if not isinstance(payload, dict) or set(payload) != {'high_level_goal', 'steps'}:
        raise ValueError('Expected exactly high_level_goal and steps')
    if not text(payload['high_level_goal']) or not isinstance(payload['steps'], list):
        raise ValueError('Invalid goal or steps')
    previous_end = 0.0
    for index, step in enumerate(payload['steps'], 1):
        keys = {'step_id', 'step_goal', 'start_sec', 'end_sec', 'evidence_status',
                'action_semantics', 'object_inventory', 'hand_object_relations'}
        if not isinstance(step, dict) or set(step) != keys:
            raise ValueError(f'Step {index}: invalid keys')
        if type(step['step_id']) is not int or step['step_id'] != index or not text(step['step_goal']):
            raise ValueError(f'Step {index}: invalid id/goal')
        start, end = step['start_sec'], step['end_sec']
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end)):
            raise ValueError(f'Step {index}: timestamps must be finite numbers')
        if not (previous_end <= start < end <= duration):
            raise ValueError(f'Step {index}: interval overlaps or is out of range')
        previous_end = end
        if step['evidence_status'] not in ('observed', 'uncertain'):
            raise ValueError(f'Step {index}: invalid evidence status')
        semantics = step['action_semantics']
        if not isinstance(semantics, dict) or set(semantics) != {'left_hand', 'right_hand', 'overall'} or not all(map(text, semantics.values())):
            raise ValueError(f'Step {index}: invalid semantics')
        inventory, relations = step['object_inventory'], step['hand_object_relations']
        if not isinstance(inventory, list) or not isinstance(relations, list):
            raise ValueError(f'Step {index}: inventory and relations must be lists')
        refs = set()
        for obj in inventory:
            if not isinstance(obj, dict) or set(obj) != {'entity_ref', 'canonical_name'} or not all(map(text, obj.values())) or obj['entity_ref'] in refs:
                raise ValueError(f'Step {index}: invalid/duplicate object')
            refs.add(obj['entity_ref'])
        for rel in relations:
            if not isinstance(rel, dict) or set(rel) != {'hand', 'object', 'relation', 'entity_ref'} or not all(map(text, rel.values())):
                raise ValueError(f'Step {index}: invalid relation')
            if rel['hand'] not in HANDS or rel['entity_ref'] not in refs:
                raise ValueError(f'Step {index}: invalid hand or unresolved object')
    return payload


def parse_response(raw):
    raw = raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = '\n'.join(raw.splitlines()[1:-1])
    return json.loads(raw)


def export(payload, manifest, run):
    fps = manifest['video']['fps']
    offset, stop = manifest['start_frame'], manifest['end_frame']
    steps = []
    previous_end = offset
    for value in payload['steps']:
        step = dict(value)
        # Round shared boundaries consistently so quantization creates no overlap.
        start = max(offset, offset + round(step['start_sec'] * fps))
        end = min(stop, offset + round(step['end_sec'] * fps))
        if start < previous_end or end <= start:
            raise ValueError('Timestamp quantization creates an empty/overlapping step; review response')
        step.update(clip_local_start_sec=step.pop('start_sec'),
                    clip_local_end_sec=step.pop('end_sec'))
        step.update(start_frame_0based=start, end_frame_0based_exclusive=end,
                    end_frame_0based_inclusive=end - 1,
                    start_sec=start / fps, end_sec=end / fps,
                    boundary_precision='vlm_approximate_quantized_to_source_frames')
        steps.append(step)
        previous_end = end
    output = {'schema': 'franka_wuji_semantic_demo.v1',
              'dataset_context': manifest,
              'high_level_goal': payload['high_level_goal'], 'steps': steps,
              'capabilities': {'step_semantics': True, 'bbox_tracking': False,
                               'depth_labels': False, 'frame_accurate_boundaries': False},
              'review_status': 'needs_human_review'}
    write_json(run / 'final_plan.json', output)
    write_json(run / 'instruction_subtask.json', {
        'instruction': payload['high_level_goal'],
        'sub_task': [step['step_goal'] for step in steps]})
    records = []
    for step in steps:
        records.append('<tr><td>' + str(step['step_id']) + '</td><td>'
                       + f'{step["start_sec"]:.3f}–{step["end_sec"]:.3f}' + '</td><td>'
                       + html.escape(step['step_goal']) + '</td><td>'
                       + html.escape(step['evidence_status']) + '</td></tr>')
    (run / 'report.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Franka/Wuji demo</title>'
        '<h1>Franka/Wuji semantic demo — human review required</h1>'
        '<p>Video playback is clip-local. Table times are episode-local.</p>'
        '<video controls width="800" src="clip.mp4"></video><p>'
        + html.escape(payload['high_level_goal'])
        + '</p><table border="1"><tr><th>Step</th><th>Episode seconds</th>'
          '<th>Goal</th><th>Evidence</th></tr>' + ''.join(records) + '</table>')
    covered = sum(s['end_frame_0based_exclusive'] - s['start_frame_0based'] for s in steps)
    write_json(run / 'validation_report.json', {
        'structural_status': 'passed', 'semantic_status': 'needs_human_review',
        'step_count': len(steps), 'covered_frames': covered,
        'uncovered_frames': stop - offset - covered,
        'note': 'Schema/time/reference checks only; does not certify visual correctness.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--camera', choices=['cam0', 'cam1', 'cam2'], default='cam0')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, help='Exclusive; defaults to episode length')
    parser.add_argument('--prepare-only', action='store_true', help='No API import, key, or network needed')
    parser.add_argument('--model', default=os.environ.get('MODEL_NAME', ''))
    parser.add_argument('--api-base', default=os.environ.get('API_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3'))
    parser.add_argument('--video-fps', type=float, default=5.0)
    parser.add_argument('--timeout', type=float, default=420.0)
    args = parser.parse_args()
    root, run = args.dataset_root.resolve(), args.run_dir.resolve()
    if root == run or root in run.parents:
        parser.error('Run directory must be outside the source dataset')
    if args.video_fps <= 0 or not math.isfinite(args.video_fps):
        parser.error('--video-fps must be positive and finite')
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            parser.error(f'{tool} is required')
    info = json.loads((root / 'meta/info.json').read_text())
    episodes = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    episode = next((ep for ep in episodes if ep['episode_index'] == args.episode), None)
    if episode is None:
        parser.error('Episode does not exist')
    camera = f'observation.images.{args.camera}'
    source = root / info['video_path'].format(
        episode_chunk=args.episode // info['chunks_size'],
        episode_index=args.episode, video_key=camera)
    end = args.end_frame if args.end_frame is not None else episode['length']
    if not (0 <= args.start_frame < end <= episode['length']):
        parser.error('Invalid episode frame interval')
    if not args.prepare_only and (not args.model or not (os.environ.get('ARK_API_KEY') or os.environ.get('API_KEY'))):
        parser.error('Set MODEL_NAME (or --model) and ARK_API_KEY (or API_KEY)')
    run.mkdir(parents=True, exist_ok=True)
    local = run / 'source.mp4'
    source_stat = source.stat()
    source_hash = digest(source)
    if not local.exists() or digest(local) != source_hash:
        temp = run / 'source.copying'
        shutil.copyfile(source, temp)
        if digest(temp) != source_hash:
            temp.unlink()
            raise ValueError('Source changed during copy or copy checksum mismatch')
        temp.replace(local)
    if (source.stat().st_size, source.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError('Source changed during preparation')
    video = probe(local)
    if video['frames'] != episode['length'] or not math.isclose(video['fps'], info['fps'], rel_tol=1e-6):
        raise ValueError('Decoded video frame count/FPS differs from metadata')
    feature = info['features'][camera]
    if [video['height'], video['width'], 3] != feature['shape']:
        raise ValueError('Video dimensions differ from metadata')
    system = (HERE / 'robot_prompt.txt').read_text()
    manifest = {'dataset_id': root.name, 'dataset_root': str(root),
                'episode_index': args.episode, 'source_recording_id': episode.get('source_recording_id'),
                'robot_type': info['robot_type'], 'camera_key': camera,
                'source_video': str(source), 'source_video_sha256': source_hash,
                'video': video, 'start_frame': args.start_frame, 'end_frame': end,
                'frame_index_base': 0, 'endpoint_semantics': 'exclusive',
                'task_context': episode.get('tasks', []),
                'prompt_sha256': hashlib.sha256(system.encode()).hexdigest(),
                'model': args.model, 'api_base': args.api_base, 'model_video_fps': args.video_fps}
    manifest_path = run / 'input_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Run configuration/source/prompt changed; choose a NEW --run-dir')
    write_json(manifest_path, manifest)
    clip = run / 'clip.mp4'
    if not clip.exists():
        temp = run / 'clip.building.mp4'
        select = f'select=between(n\\,{args.start_frame}\\,{end - 1}),setpts=N/({video["fps"]}*TB)'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(local), '-map', '0:v:0',
                        '-vf', select, '-fps_mode', 'passthrough', '-c:v', 'libx264',
                        '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-an', str(temp)], check=True)
        temp.replace(clip)
    clip_meta = probe(clip)
    if clip_meta['frames'] != end - args.start_frame or not math.isclose(clip_meta['fps'], video['fps'], rel_tol=1e-6):
        raise ValueError('Prepared clip frame count/FPS mismatch')
    duration = (end - args.start_frame) / video['fps']
    user = (f'Clip duration: {duration:.9f} seconds; source FPS: {video["fps"]}. '
            f'Return clip-local seconds in [0, {duration:.9f}]. '
            f'Robot type: {info["robot_type"]}. Camera: {camera}; wrist side is unverified. '
            f'Task context (not ground truth): {json.dumps(episode.get("tasks", []))}')
    (run / 'system_prompt.txt').write_text(system)
    (run / 'user_prompt.txt').write_text(user)
    summary = {'status': 'prepared', 'network_calls_this_run': 0,
               'clip_frames': clip_meta['frames'], 'clip_seconds': duration}
    write_json(run / 'run_summary.json', summary)
    if args.prepare_only:
        print(json.dumps(summary, indent=2)); return
    cached = run / 'model_response.json'
    if cached.exists():
        payload = validate_response(json.loads(cached.read_text()), duration)
        usage = json.loads((run / 'usage.json').read_text()) if (run / 'usage.json').exists() else {}
    else:
        sys.path.insert(0, str(PIPELINE))
        try:
            from api_client_qwen_v2 import create_api_client, build_request_payload_input, call_model_api
        except ImportError as exc:
            raise RuntimeError('Install demos/franka_wuji/requirements.txt into your demo environment') from exc
        client = create_api_client(args.api_base, os.environ.get('ARK_API_KEY') or os.environ['API_KEY'])
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': [
            {'type': 'input_text', 'text': user},
            {'type': 'input_video_path', 'video_path': str(clip), 'fps': args.video_fps}]}]
        write_json(run / 'run_summary.json', {**summary, 'status': 'calling_model'})
        try:
            raw, usage = call_model_api(client, model_name=args.model,
                payload_input=build_request_payload_input(messages), timeout_sec=args.timeout,
                video_fps=args.video_fps, max_output_tokens=8000, reasoning_effort='medium')
            (run / 'raw_response.txt').write_text(raw)
            write_json(run / 'usage.json', usage)
            payload = validate_response(parse_response(raw), duration)
            # Cache only validated responses. Failed validation is never completed.
            write_json(cached, payload)
        except Exception as exc:
            write_json(run / 'run_summary.json', {**summary, 'status': 'failed',
                       'error_type': type(exc).__name__, 'error': str(exc)})
            raise
        summary['network_calls_this_run'] = 1
    try:
        export(payload, manifest, run)
    except Exception as exc:
        write_json(run / 'run_summary.json', {**summary, 'status': 'failed',
                   'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    write_json(run / 'run_summary.json', {**summary, 'status': 'completed',
               'token_usage': usage, 'review_status': 'needs_human_review'})
    print(f'Completed: {run / "final_plan.json"}\nReview: {run / "report.html"}')


if __name__ == '__main__':
    main()
