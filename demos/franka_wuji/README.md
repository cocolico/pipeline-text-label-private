# Franka + Wuji: episode_000000 semantic annotation demo

This standalone demo reuses `api_client_qwen_v2.py` for Ark video understanding.
It does not invoke the original Stage 1–3 pipeline. It needs neither upstream
hand screening nor SAM3/WiLoR, GPU, depth, PyArrow, or the cleaning UI.
The source dataset is read-only; all generated files go to `--run-dir` outside it.
It does not upload annotations to the remote data server. An annotation run DOES
upload its selected clip to the configured model service. Prepare-only uses no network.

## Current machine

```bash
cd /home/ty/ty_works/pipeline_text_label-20261016-172802/pipeline_text_label
export DATASET_ROOT='/home/ty/company-remote-2/real/harvest-tomato/data/data-lerobot/wuji/A/209_213_data_lerobot'
export DEMO_ROOT='/home/ty/ty_works/annotation_runs/franka_wuji'
# Use a model ID that your Ark account actually authorizes. This value is part of the run fingerprint.
export MODEL_NAME='YOUR_ACCESSIBLE_ARK_VIDEO_MODEL'
```

Required system programs: `python3`, `ffmpeg`, `ffprobe`.
Create a new environment; do not reuse the copied `.venv_pipeline` or its old ffmpeg symlink.

```bash
python3 -m venv /home/ty/ty_works/.venv-franka-wuji-demo
/home/ty/ty_works/.venv-franka-wuji-demo/bin/python -m pip install -r demos/franka_wuji/requirements.txt
```

First verify a 30-second exact-frame clip (frames [0, 900)); this is a test window,
not an assertion that the first 30 seconds form one semantic step.

```bash
python3 demos/franka_wuji/run_demo.py \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "$DEMO_ROOT/ep000000_cam0_first30s" \
  --episode 0 --camera cam0 --start-frame 0 --end-frame 900 \
  --prepare-only
```

Prepare-only copies only the chosen camera's full source video locally, checks
SHA256, decodes/counts the source and selected clip, compares dimensions/FPS/frame
count with metadata, then saves manifest and prompts. It does not inspect Parquet
rows or prove multimodal alignment. For episode 0 expected source: 3520 frames,
30 FPS, 640x400; first clip: 900 frames, 30 seconds.

Configure credentials without embedding them in source or command history:

```bash
read -rsp 'ARK_API_KEY: ' ARK_API_KEY
echo
export ARK_API_KEY
export API_BASE_URL='https://ark.cn-beijing.volces.com/api/v3'
```

Run the same command without `--prepare-only`, with the new interpreter:

```bash
/home/ty/ty_works/.venv-franka-wuji-demo/bin/python demos/franka_wuji/run_demo.py \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "$DEMO_ROOT/ep000000_cam0_first30s" \
  --episode 0 --camera cam0 --start-frame 0 --end-frame 900
```

There is no automatic retry. Inspect raw_response.txt and run_summary.json on
failure; an invalid response is not completed/cached, and rerunning may incur a
new model charge. A validated cached response is reused without another model
call. Keep credentials set even when using cache. An interrupted call before its
response is persisted cannot be safely deduplicated by this prototype.

After inspecting the short result, run the complete episode in a NEW run directory:

```bash
/home/ty/ty_works/.venv-franka-wuji-demo/bin/python demos/franka_wuji/run_demo.py \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "$DEMO_ROOT/ep000000_cam0_full" \
  --episode 0 --camera cam0 --end-frame 3520
```

Model ID, endpoint, sampling FPS, prompt, source fingerprint or interval changes
require a NEW run directory. Set MODEL_NAME BEFORE preparation and keep it unchanged
when proceeding to annotation. Use a new run directory if preparation used an
empty model ID. Use --video-fps 5 for the initial trial; full episode is about 586.7
sampling points at 5 FPS and may exceed some models' limits. If the service rejects
it or misses fast actions, use explicit shorter windows in separate runs, preserving
--start-frame offsets; this script does not automatically merge window annotations.

## Outputs

- source.mp4: unmodified local copy of the selected source video.
- clip.mp4: exact decoded-frame interval, locally reencoded.
- input_manifest.json: source SHA256, episode, camera, interval, prompt and model configuration.
- system_prompt.txt / user_prompt.txt: actual prompts.
- raw_response.txt / usage.json: original response and usage, after a model call.
- model_response.json: validated model schema, clip-local seconds.
- final_plan.json: semantic steps, object inventory, hand-object relations, source-frame mapping.
- instruction_subtask.json: compact language export.
- validation_report.json: structural validation and coverage, NOT visual certification.
- report.html: local clip player plus step table for human review.
- run_summary.json: preparation/model/export status.

Open report.html locally. If file video playback is restricted, serve only the
run directory bound to localhost, using a separate terminal:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory "$DEMO_ROOT/ep000000_cam0_first30s"
```

Then open http://127.0.0.1:8765/report.html. Video player time is clip-local;
table times are episode-local. Inspect both raw semantics and actual video.

## Boundary and quality limits

The model predicts approximate clip-local seconds. The exporter maps those to
original episode frame indices with offset + round(seconds * FPS), exclusive end.
This arithmetic is tested; the visual event boundary itself is NOT frame-accurate.
No bboxes, depth, arm visibility, cross-view identity, or Parquet-row validation is
claimed. Left/right may remain unknown_hand. Do not assign wrist side from cam1/2
names. The existing task instruction is context only; failure/uncertainty must not
be rewritten as task success. Objects are step-local, not cross-step identities.
Empty output can pass structural validation but is not a successful useful annotation.

The original pipeline requires real hand screening + robot prompt adaptation for
full integration. This demo first checks video access, model compatibility,
semantic quality, JSON export and frame-offset arithmetic independently.

## Offline tests

```bash
python3 -m unittest discover -s demos/franka_wuji -p test_demo.py -v
```

On an industrial PC, deploy the project and dependencies, replace DATASET_ROOT
with its LOCAL cleaned dataset path, and keep DEMO_ROOT on local writable storage.
No remote processing/UI integration is part of this demo.
