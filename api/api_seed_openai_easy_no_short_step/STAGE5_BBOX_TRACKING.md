# Stage 5: Per-frame hand and manipulated-object boxes

Stage 5 is optional. The default pipeline remains `--stages 1,2,3,4`, so existing
Stage 1-4 behavior is unchanged.

## Setup

```bash
./setup_stage5_bbox_env.sh
```

The script prepares the WiLoR Stage 5 environment and validates the separate SAM3
environment/checkpoint. Grounding DINO and SAM2 remain available only for the explicit
legacy compatibility path. WiLoR may run on CPU without hiding CUDA from the isolated
SAM3 subprocess.

## Run

Run all five stages:

```bash
python pipeline.py \
  --input-video /absolute/path/video.mp4 \
  --output-root /absolute/path/output \
  --stages 1,2,3,4,5 \
  --bbox-python /Users/lululzz/Desktop/litchi/api/.venv_bbox_tracking/bin/python
```

Run only Stage 5 for an output that already has `final_plan.json`:

```bash
python pipeline.py \
  --input-video /absolute/path/video.mp4 \
  --output-root /absolute/path/output \
  --stages 5 \
  --bbox-python /Users/lululzz/Desktop/litchi/api/.venv_bbox_tracking/bin/python
```

Use `--no-retry-failed --overwrite` to force regeneration. Use
`--no-api-grounding` to prohibit Ark fallback. By default, unresolved manipulated
objects fail Stage 5 instead of silently producing an incomplete annotation.

## Inputs and identity

Stage 5 reads Stage 4 `final_plan.json`. Every step must contain the Stage 3
`hand_object_relations` array; the array may be empty only when the primary operator
has no visibly supported hand-object relation. Those relations define which objects
are task-relevant and which anatomical hand should be associated with each object.

Conservative label alias matching keeps the same physical object ID across steps,
while contradictory color, material, side, or ordinal descriptors remain separate.
WiLoR class labels (`left`, `right`) are preserved; image position never swaps
anatomical left/right identities. When multiple people are visible, only the
temporally stable hand pair operating closest to the image center is retained.

## Detection policy

1. WiLoR detects all left/right-hand candidates on every decoded original frame.
2. Center-pair scoring plus a hard short-horizon motion gate retains one primary operator.
3. SAM3 tracks hands and objects. Each anatomical hand receives one primary instance
   chain through sequence-wide anchor assignment. A non-overlapping fragment can join
   that chain only after strict boundary continuity validation; only one instance may
   be active per hand per frame, and unselected edge instances cannot replace it.
4. Ark image grounding is used conditionally for sustained weak/missing primary hands
   or weak/unresolved object tracks. Hand rescue receives previous/target/next images;
   every returned anchor is checked against local temporal, primary-hand, SAM3, size,
   and hand-duplicate evidence. Structured `not_visible` responses annotate only a
   bounded window of at most 0.5 seconds and require the explicit reason `occluded`,
   `out_of_frame`, or `absent`; `unknown` cannot suppress a missing-track warning.
5. Weak object tracks receive interval-scoped optical-flow completion. Grounding DINO
   plus SAM2 is used only when `--sam3-legacy-fallback` is explicitly enabled.

Near-identical simultaneous left/right boxes are relation-resolved: a hand explicitly
expected by the current step is retained; otherwise the stronger direct evidence is
retained and the duplicate label is left without a box.

Partially visible objects receive an estimated amodal box. Fully hidden or absent
objects have `bbox=null` and an explicit visibility state.

## Outputs

Files are written under `<video_output>/stage5/`:

- `frame_bboxes.jsonl`: one record for every original frame, starting at frame 0.
- `tracks.json`: stable hand/object IDs, aliases, step membership, and object anchors.
- `qa_summary.json`: primary-operator and SAM3 instance assignment, per-track active
  coverage/gaps, center/IoU/area/instance discontinuities, unresolved objects, model
  errors, API rejection reasons, and automatic-use eligibility.
- `api_usage.json`: Ark calls, cache hits, and token usage.
- `bbox_overlay.mp4`: optional 30fps visual audit video.
- `stage5_summary.json`: source/config hashes and completion metadata.

Resume accepts only outputs whose quality gate is `passed` and explicitly eligible for
automatic use; warning and failed outputs are regenerated.

Each visible entity has both integer pixel `xyxy` coordinates and official normalized
`xyxy` coordinates in `[0,999]`. Frame timestamps use ffprobe presentation timestamps
normalized to frame 0 when available; `frame_index / fps` is the explicit fallback.

Ark coordinate conversion follows the official Grounding specification:
https://docs.volcengine.com/docs/82379/1616136?lang=zh
